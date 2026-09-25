"""SATYAVAANI - the live call engine.
Every HOP_SEC it takes the last 3 s of the call and does ONE wav2vec2 pass; the same features go to
  * the deepfake head  -> p_synth  (is this voice synthetic?)
  * the speaker head   -> voiceprint -> cosine to the person the caller claims to be
and fuses them into one risk score:
  risk = P(synthetic OR not the claimed person), shifted by the call-context prior, smoothed over time.
No torch in this file: the models are passed in, so the logic is testable on its own.
"""
import json, time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from sv_audio import SR, WIN, NAME, REPORTS, TRAIN_HOP, speech_fraction, trim_silence, rms_norm, pad_to, telephonize
from sv_heads import sigmoid, logit, load_voiceprints, save_voiceprints, voiceprint

HOP_SEC            = 3.0    # a new verdict every 3 s
MIN_SPEECH_FRAC    = 0.4    # skip windows that are mostly silence
MIN_SPK_SPEECH_SEC = 4.0    # identity is only judged after 4 s of speech
K_SPK              = 15.0   # how sharply "voice doesn't match" turns into risk around the threshold
EMA_KEEP           = 0.5    # smoothing of the risk (in log-odds) from window to window
PASS_BELOW, HOLD_AT = 45.0, 75.0
CONTEXT_PRIORS = OrderedDict([("Unknown number", 0.7), ("Asks for money / OTP", 1.0), ("Urgent or secret request", 0.5)])

def verdict_of(risk):
    return "PASS" if risk < PASS_BELOW else ("VERIFY" if risk < HOLD_AT else "HOLD")

def fuse(p_synth, cos, thr, prior, have_spk):
    """P(threat) = 1 - (1 - p_synth)(1 - p_imp). A voice match can only leave p_imp near 0, never lower risk."""
    p_imp = float(sigmoid(K_SPK * (thr - cos))) if have_spk else 0.0
    p = 1.0 - (1.0 - p_synth) * (1.0 - p_imp)
    return float(logit(p) + prior), p_imp

@dataclass
class CallState:
    claimed: str = None          # voiceprint id the caller claims to be (None = unknown caller)
    source: str = None           # demo only: whose voice is really in the audio -> the fold head that never saw them
    prior: float = 0.0
    phone: bool = False           # live audio: send each window through the phone line before scoring
    n: int = 0                   # samples received
    buf: np.ndarray = field(default_factory=lambda: np.zeros(0, "float32"))
    buf_start: int = 0
    next_end: int = WIN
    L_ema: float = None
    emb_sum: np.ndarray = None
    speech_sec: float = 0.0
    risk: float = 0.0
    verdict: str = "LISTENING"
    peak: float = 0.0
    alert: str = ""
    history: list = field(default_factory=list)
    p_recent: list = field(default_factory=list)

class Engine:
    def __init__(self, featurize, deepfake_logit, speaker_embed, spk_threshold, calib,
                 voiceprints=None, info=None):
        self.featurize, self.deepfake_logit, self.speaker_embed = featurize, deepfake_logit, speaker_embed
        self.thr = float(spk_threshold)
        self.cal_a, self.cal_b = float(calib.get("a", 1.0)), float(calib.get("b", 0.0))
        self.voiceprints = voiceprints if voiceprints is not None else OrderedDict()
        self.info = info or {}

    # ------------------------------------------------------------------ real models
    @classmethod
    def load(cls, verbose=True):
        import torch, sv_models as M
        if verbose: print("loading wav2vec2 + heads ...")
        enc = M.Encoder()
        bank, cal = M.load_deepfake_bank()
        if not bank:
            raise SystemExit("No deepfake heads in models\\deepfake\\ - run step4_deepfake.py first.")
        spk = M.load_speaker_head(verbose=verbose)
        vps = load_voiceprints()
        rep = lambda n: json.loads((REPORTS / n).read_text()) if (REPORTS / n).exists() else {}
        spk_rep, df_rep, lat = rep("speaker_test.json"), rep("deepfake_loso.json"), rep("latency.json")
        thr = spk_rep.get("threshold_used", spk.threshold)

        def featurize(wins): return enc(wins)
        def df_logit(feats, source): return float((bank.get(source) or bank["all"]).logits(feats).mean().item())
        def spk_embed(feats): return spk.embed(feats.detach().cpu().numpy())

        info = {"device": enc.dev, "gpu": torch.cuda.get_device_name(0) if enc.dev == "cuda" else "CPU",
                "df_layers": df_rep.get("layer_weights") or list(map(float, bank["all"].layer_weights())),
                "spk_layers": list(map(float, spk.layer_weights())), "spk_arch": spk.arch,
                "deepfake": df_rep, "speaker": spk_rep, "latency": lat, "folds": sorted(k for k in bank if k != "all")}
        eng = cls(featurize, df_logit, spk_embed, thr, cal, vps, info)
        eng.info["startup_ms"] = eng.warmup()
        if verbose:
            print(f"ready on {info['gpu']}: {eng.info['startup_ms']:.1f} ms per 3 s window, "
                  f"{len(vps)} voiceprints, speaker threshold {thr:.3f}")
        return eng

    def warmup(self, n=8):
        w = (np.random.default_rng(0).standard_normal(WIN) * 0.05).astype("float32")
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            f = self.featurize([w]); self.deepfake_logit(f, None); self.speaker_embed(f)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts[2:])) if n > 2 else float(ts[-1])

    # ------------------------------------------------------------------ calls
    def new_call(self, claimed=None, source=None, prior=0.0, phone=False):
        return CallState(claimed=claimed if claimed in self.voiceprints else None, source=source,
                         prior=float(prior), phone=bool(phone))

    def push(self, st, chunk):
        """Feed new audio (16 kHz float32). Scores every complete window. Returns the new window records."""
        chunk = np.asarray(chunk, dtype="float32").ravel()
        st.buf = np.concatenate([st.buf, chunk]); st.n += len(chunk)
        hop, out = int(HOP_SEC * SR), []
        while st.n >= st.next_end:
            e = st.next_end - st.buf_start
            out.append(self._score(st, st.buf[e - WIN: e], st.next_end / SR))
            st.next_end += hop
            keep_from = st.next_end - WIN
            if keep_from > st.buf_start:
                st.buf = st.buf[keep_from - st.buf_start:]; st.buf_start = keep_from
        return out

    def _name(self, vid):
        return self.voiceprints[vid][0] if vid in self.voiceprints else NAME.get(vid, str(vid))

    def _score(self, st, w, t):
        frac = speech_fraction(w)
        rec = {"t": round(t, 2), "speech": round(frac, 2), "scored": False}
        if frac < MIN_SPEECH_FRAC:
            rec.update(risk=st.risk, verdict=st.verdict, reason="Listening - too little speech in this window")
            st.history.append(rec); return rec

        t0 = time.perf_counter()
        if st.phone: w = telephonize(w)                           # live mic: 8 kHz G.711, like a real call
        feats = self.featurize([w])                               # ONE encoder pass ...
        z = self.deepfake_logit(feats, st.source)                 # ... deepfake head
        emb = np.asarray(self.speaker_embed(feats))[0]            # ... speaker head, same features
        ms = (time.perf_counter() - t0) * 1000
        p_now = float(sigmoid(self.cal_a * z + self.cal_b))
        st.p_recent = (st.p_recent + [p_now])[-3:]
        p_synth = float(np.median(st.p_recent))    # one odd window (shout, laugh, shock) can't swing the call

        st.speech_sec += frac * HOP_SEC
        st.emb_sum = emb.copy() if st.emb_sum is None else st.emb_sum + emb
        avg = st.emb_sum / (np.linalg.norm(st.emb_sum) + 1e-9)
        sims = {vid: float(avg @ e) for vid, (_, e) in self.voiceprints.items()}
        best = max(sims, key=sims.get) if sims else None
        cos = sims.get(st.claimed) if st.claimed else None
        have_spk = cos is not None and st.speech_sec >= MIN_SPK_SPEECH_SEC

        L, p_imp = fuse(p_synth, cos if cos is not None else 1.0, self.thr, st.prior, have_spk)
        st.L_ema = L if st.L_ema is None else EMA_KEEP * st.L_ema + (1 - EMA_KEEP) * L
        st.risk = float(100 * sigmoid(st.L_ema)); st.verdict = verdict_of(st.risk); st.peak = max(st.peak, st.risk)

        voice = f"Synthetic voice {p_synth:.0%}" if p_synth >= 0.5 else f"Human voice ({1 - p_synth:.0%} sure)"
        if st.claimed is None:
            who = f"Unknown caller - sounds most like {self._name(best)} ({sims[best]:.2f})" if best else "Unknown caller"
        elif not have_spk:
            who = f"Checking it's {self._name(st.claimed)} ... {st.speech_sec:.1f}/{MIN_SPK_SPEECH_SEC:.0f} s of speech"
        elif cos >= self.thr:
            who = f"Voice matches {self._name(st.claimed)} ({cos:.2f})"
        else:
            who = f"Voice does NOT match {self._name(st.claimed)} ({cos:.2f} < {self.thr:.2f})"
        if st.verdict == "HOLD" and not st.alert:
            st.alert = ("Likely AI-cloned voice. Hold any payment and call back on a saved number."
                        if p_synth >= p_imp else
                        f"This is not {self._name(st.claimed)}'s voice. Verify identity before acting.")
        rec.update(scored=True, p_synth=round(p_synth, 4), p_window=round(p_now, 4), p_imp=round(p_imp, 4),
                   cos=None if cos is None else round(cos, 3), best=best,
                   best_cos=None if best is None else round(sims[best], 3),
                   risk=round(st.risk, 1), verdict=st.verdict, reason=f"{voice} | {who}", ms=round(ms, 1))
        st.history.append(rec)
        return rec

    # ------------------------------------------------------------------ whole files (API) + enrolment
    def score_audio(self, w, claimed=None, source=None, prior=0.0):
        w = np.asarray(w, "float32")
        if len(w) < WIN: w = pad_to(w, WIN)
        st = self.new_call(claimed, source, prior)
        self.push(st, w)
        scored = [r for r in st.history if r["scored"]]
        return {"claimed_id": st.claimed, "claimed_name": self._name(st.claimed) if st.claimed else None,
                "seconds": round(len(w) / SR, 2), "windows": st.history,
                "final_risk": round(st.risk, 1), "peak_risk": round(st.peak, 1), "verdict": st.verdict,
                "alert": st.alert or None,
                "deepfake_prob_mean": round(float(np.mean([r["p_synth"] for r in scored])), 4) if scored else None,
                "speaker_cos": scored[-1]["cos"] if scored else None}

    def voiceprint_of(self, w):
        """Every speech window of a recording, sent through a phone line (8 kHz G.711) and embedded, then
        averaged: the voiceprint is a phone-call voiceprint, matching how calls (and the mic with the
        phone-line box ticked) are scored."""
        w = pad_to(rms_norm(trim_silence(np.asarray(w, "float32"))), WIN)
        starts = range(0, len(w) - WIN + 1, TRAIN_HOP)
        keep = [s for s in starts if speech_fraction(w[s: s + WIN]) >= MIN_SPEECH_FRAC] or [0]
        tel = telephonize(w)
        wins = [tel[s: s + WIN] for s in keep]          # phone-line (codec) copy only
        embs = np.concatenate([np.asarray(self.speaker_embed(self.featurize(wins[i:i + 8])))
                               for i in range(0, len(wins), 8)])
        return voiceprint(embs), len(keep)

    def enroll(self, name, w, vid=None):
        vp, n = self.voiceprint_of(w)
        if vid is None:
            k = 1
            while f"u{k:02d}" in self.voiceprints: k += 1
            vid = f"u{k:02d}"
        self.voiceprints[vid] = (str(name), vp)
        save_voiceprints(self.voiceprints)
        return vid, n
