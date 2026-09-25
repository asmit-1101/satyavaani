"""Train the SPEAKER head two ways and test each on matching audio - vs clones too.
    python -u train_speaker_variants.py                 (needs libri_generalization.py run first: it made the clones)
    python -u train_speaker_variants.py --only mic      (or --only codec)
Variants (same recipe for training AND testing: pauses removed and filled with noise, random room echo,
background noise 5-25 dB):
  mic   : that audio as is            -> train mic,   test mic
  codec : that audio + phone line     -> train codec, test codec
  mix   : your CURRENT head (trained on clean + phone copies), tested on both kinds pooled (for comparison)
Tests:
  LibriSpeech - the 20 TEST speakers of libri_exp (never trained on): genuine vs other speakers,
                + how many XTTS clones and kNN-VC fakes of them pass the voice check
  Your team   - 6 teammates (never trained on): enrol 001-010, test 011-030, + their XTTS clones
Saves libri_exp\\spk_head_mic.pt, libri_exp\\spk_head_codec.pt, libri_exp\\speaker_variants.json.
Your app's models\\spk_head.pt is NOT touched.
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from sv_audio import (SR, WIN, SPEAKERS, ENROL_CLIPS, TEST_CLIPS, real_clip, xtts_clip, load16, rms_norm,
                      trim_silence, pad_to, train_windows, gate_pauses, add_noise, telephonize, rng_for, to16k)
from sv_heads import eer, voiceprint, SpeakerNP, split_checkpoint
from sv_models import Encoder, load_speaker_head, DEV
import libri_generalization as LG                      # reuses its manifest, clone folders and room() echo

CLIPS_PER_SPK, WINS_PER_CLIP, N_VAL = 15, 2, 20
SNR = (5.0, 25.0)
EPOCHS, BS = 80, 128
OUT = LG.EXP / "speaker_variants.json"

def process(w, key, channel):
    """pauses gated -> room echo -> noise -> [phone line]; identical for every class"""
    rng = rng_for(key)
    c = pad_to(rms_norm(gate_pauses(rms_norm(w))), WIN)
    x = add_noise(rms_norm(LG.room(c, rng)), rng, snr_db=SNR)
    return telephonize(x) if channel == "codec" else x

def load_any(p):
    return rms_norm(trim_silence(load16(p)))

# ------------------------------------------------------------------ model (same design as the original head)
class AAM(nn.Module):
    def __init__(self, E, n, m=0.25, s=30.0):
        super().__init__(); self.W = nn.Parameter(torch.randn(n, E) * 0.01); self.m, self.s = m, s
    def forward(self, e, y):
        cos = F.linear(F.normalize(e), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        return F.cross_entropy(self.s * torch.where(F.one_hot(y, cos.shape[1]).bool(), torch.cos(torch.acos(cos) + self.m), cos), y)

class SpeakerHead(nn.Module):
    def __init__(self, n, L=13, D=1536):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(L))
        self.net = nn.Sequential(nn.Dropout(0.2), nn.LayerNorm(D), nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 192))
        self.aam = AAM(192, n)
    def embed(self, x): return self.net((torch.softmax(self.alpha, 0)[None, :, None] * x).sum(1))

# ------------------------------------------------------------------ training data
def libri_train_speakers(man):
    libri = LG.find_libri()
    test = {c["spk"] for c in man if c["split"] == "test"}
    spks = sorted(p for p in libri.iterdir() if p.is_dir() and p.name.isdigit() and p.name not in test)
    return libri, spks

def train_features(enc, man, channel):
    cache = LG.EXP / f"spk_train_{channel}.npz"
    if cache.exists():
        d = np.load(cache); return d["X"], d["y"]
    libri, spks = libri_train_speakers(man)
    rng = np.random.default_rng(1)
    X, y, buf, lab = [], [], [], []
    for i, sp in enumerate(spks):
        files = sorted(sp.glob("*/*.flac")); rng.shuffle(files); got = 0
        for f in files:
            if got >= CLIPS_PER_SPK: break
            w, sr = sf.read(str(f), dtype="float32", always_2d=True); w = to16k(w.mean(1), sr)
            if len(w) < WIN: continue
            x = process(rms_norm(trim_silence(w - w.mean())), f"spk|{channel}|{f.stem}", channel)
            starts = [0, len(x) - WIN] if len(x) >= 2 * WIN else [(len(x) - WIN) // 2]
            for s in starts[:WINS_PER_CLIP]: buf.append(x[s: s + WIN]); lab.append(i)
            got += 1
        if len(buf) >= 240:
            X.append(enc.numpy(buf, label=f"[{channel}] train windows")); y += lab; buf, lab = [], []
        print(f"\r  [{channel}] read {i + 1}/{len(spks)} speakers", end="", flush=True)
    if buf: X.append(enc.numpy(buf, label=f"[{channel}] train windows")); y += lab
    print()
    X, y = np.concatenate(X), np.array(y)
    np.savez(cache, X=X, y=y)
    return X, y

def pair_eer(E, y):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = E @ E.T; iu = np.triu_indices(len(y), 1)
    return eer(S[iu], (y[:, None] == y[None, :])[iu].astype(int))

def train_head(X, y, channel):
    ids = np.unique(y); rng = np.random.default_rng(0)
    val = set(rng.choice(ids, N_VAL, replace=False).tolist())
    tr = np.array([v not in val for v in y]); remap = {s: i for i, s in enumerate(sorted(set(y[tr].tolist())))}
    Xtr = torch.from_numpy(X[tr].astype("float32")).to(DEV); ytr = torch.tensor([remap[v] for v in y[tr]], device=DEV)
    Xva = torch.from_numpy(X[~tr].astype("float32")).to(DEV); yva = y[~tr]
    torch.manual_seed(0)
    m = SpeakerHead(len(remap)).to(DEV)
    opt = torch.optim.AdamW([{"params": [m.alpha], "lr": 1e-1, "weight_decay": 0.0},
                             {"params": list(m.net.parameters()) + list(m.aam.parameters()), "lr": 5e-4, "weight_decay": 5e-3}])
    best, state, thr, bad = 1.0, None, 0.3, 0
    print(f"  [{channel}] training on {len(ytr)} windows / {len(remap)} speakers, validating on {N_VAL} other speakers")
    for ep in range(1, EPOCHS + 1):
        m.train()
        for b in torch.randperm(len(ytr), device=DEV).split(BS):
            loss = m.aam(m.embed(Xtr[b]), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 2 == 0:
            m.eval()
            with torch.no_grad(): e, t = pair_eer(m.embed(Xva).cpu().numpy(), yva)
            if e < best: best, thr, bad, state = e, t, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else: bad += 1
            print(f"    epoch {ep:3d}  validation EER {100 * e:5.2f}%" + ("  <- best" if bad == 0 else ""))
            if bad >= 8: break
    path = LG.EXP / f"spk_head_{channel}.pt"
    torch.save({"state": state, "thr": float(thr), "val_eer": float(best), "trained_on": f"LibriSpeech, {channel}, gated+echo+noise"}, str(path))
    print(f"  [{channel}] best validation EER {100 * best:.2f}%  -> saved {path.name}")
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    st, ex = split_checkpoint(ck)
    return SpeakerNP({k: v.numpy() for k, v in st.items()}, ex, verbose=False)

# ------------------------------------------------------------------ evaluation (matched audio)
def evaluate(enc, head, man, channels, label):
    def emb(path, key):
        w = load_any(path)
        vs = [voiceprint(head.embed(enc(train_windows(process(w, f"{key}|{ch}", ch))).cpu().numpy())) for ch in channels]
        return vs                                         # one vector per channel
    def score(vps, tests):
        gen, imp, clo = [], [], {}
        for owner, kind, vecs in tests:
            for v in vecs:
                for s, vp in vps.items():
                    c = float(vp @ v)
                    if kind == "real": (gen if owner == s else imp).append(c)
                    elif owner == s: clo.setdefault(kind, []).append(c)
        gen, imp = np.array(gen), np.array(imp)
        e, t = eer(np.r_[gen, imp], np.r_[np.ones(len(gen)), np.zeros(len(imp))])
        r = dict(eer=e, threshold=t, genuine=float(gen.mean()), other=float(imp.mean()))
        for k, v in clo.items(): r[f"{k}_pass"] = float((np.array(v) >= t).mean()); r[f"{k}_cos"] = float(np.mean(v))
        return r
    res = {}
    # LibriSpeech test speakers: enrol on their reference (other chapter), test on their 8 sentences
    test = [c for c in man if c["split"] == "test"]
    vps, tests = {}, []
    for c in test:
        vps[c["spk"]] = voiceprint(np.stack(sum([emb(r, f"enr|{Path(r).stem}") for r in c["refs"]], [])))
        for u in c["utts"]:
            tests.append((c["spk"], "real", emb(u["path"], f"tst|{u['uid']}")))
            xp = LG.XDIR / f"{u['uid']}.wav"
            if xp.exists(): tests.append((c["spk"], "xtts", emb(xp, f"xt|{u['uid']}")))
    for c, u, out in LG.knn_pairs(man):
        if out.exists(): tests.append((c["spk"], "knnvc", emb(out, f"kn|{out.stem}")))
    res["librispeech"] = score(vps, tests)
    # your team
    vps, tests = {}, []
    for s, *_ in SPEAKERS:
        en = [v for k in ENROL_CLIPS if real_clip(f"{s}_{k:03d}").exists() for v in emb(real_clip(f"{s}_{k:03d}"), f"te|{s}{k}")]
        if not en: continue
        vps[s] = voiceprint(np.stack(en))
        for k in TEST_CLIPS:
            cid = f"{s}_{k:03d}"
            if real_clip(cid).exists(): tests.append((s, "real", emb(real_clip(cid), f"tt|{cid}")))
            if xtts_clip(cid).exists(): tests.append((s, "xtts", emb(xtts_clip(cid), f"tx|{cid}")))
    res["team"] = score(vps, tests) if vps else {}
    print(f"  [{label}] evaluated")
    return res

def main():
    man = LG.select()
    if not any(c["split"] == "test" for c in man): sys.exit("run libri_generalization.py first (it makes the manifest and clones)")
    enc = Encoder()
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for ch in ("mic", "codec"):
        if only and ch != only: continue
        X, y = train_features(enc, man, ch)
        head = train_head(X, y, ch)
        results[f"{ch} -> {ch}"] = evaluate(enc, head, man, (ch,), ch)
    if not only or only == "mix":
        results["current head, mix -> mix"] = evaluate(enc, load_speaker_head(verbose=False), man, ("mic", "codec"), "mix")
    OUT.write_text(json.dumps(results, indent=2))
    pc = lambda r, k: "  n/a" if not r or r.get(k) is None else f"{100 * r[k]:5.1f}%"
    print("\n" + "=" * 110)
    print("SPEAKER HEAD VARIANTS - matched audio; pauses removed+noise, room echo, 5-25 dB noise on everything")
    print("=" * 110)
    print(f"{'train -> test':<26}| {'LIBRISPEECH 20 unseen speakers':^42} | {'YOUR TEAM (never trained on)':^30}")
    print(f"{'':<26}| {'EER':>6} {'gap':>5} {'XTTS pass':>10} {'kNN-VC pass':>12} | {'EER':>6} {'gap':>5} {'XTTS pass':>10}")
    for name, r in results.items():
        L, T = r.get("librispeech", {}), r.get("team", {})
        gl = f"{L['genuine'] - L['other']:5.2f}" if L else "  n/a"; gt = f"{T['genuine'] - T['other']:5.2f}" if T else "  n/a"
        print(f"{name:<26}| {pc(L, 'eer')} {gl} {pc(L, 'xtts_pass'):>10} {pc(L, 'knnvc_pass'):>12} | {pc(T, 'eer')} {gt} {pc(T, 'xtts_pass'):>10}")
    print("\nEER: lower = tells people apart better.  gap: real-person minus other-person similarity (bigger = better).")
    print("pass: % of fakes the voice check accepts at the EER threshold - high means the voice check alone is fooled.")
    print(f"saved {OUT.name}; new heads in libri_exp\\ - models\\spk_head.pt (your app) untouched")

if __name__ == "__main__":
    main()
