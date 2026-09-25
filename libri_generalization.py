"""LibriSpeech generalization experiment - standalone, touches NOTHING in data\\ models\\ reports\\.
    python -u libri_generalization.py                       (all stages, resumes where it stopped)
    python -u libri_generalization.py --libri "D:\\...\\train-clean-100"
    python -u libri_generalization.py --stage eval          (only retrain + evaluate)
    python -u libri_generalization.py --stage eval --configs C,D   (only some noise settings)
Question: a deepfake head trained on XTTS clones only - does it catch kNN-VC voice conversion?
  1 select  : N_SPK LibriSpeech speakers, N_UTT sentences each (4-12 s), reference audio from ANOTHER chapter
  2 xtts    : XTTS says the SAME sentence in the SAME speaker's voice (reference = other chapter)
  3 knnvc   : test speakers only - another test speaker's sentence converted into this speaker's voice
  4 features: identical recipe for every class, NO pitch/speed change: pauses gated and filled with noise,
              optional random room echo, mic tilt, background noise at the setting's SNR, then EVERYTHING
              through the phone line (8 kHz G.711); 3 s windows, wav2vec2 13 layers.
              Settings A-D (CONFIGS) are run one after another and compared in one table.
  5 eval    : train on FIT speakers (real vs XTTS), calibrate on CAL speakers, test on unseen TEST speakers:
              real vs XTTS (seen generator) and real vs kNN-VC (unseen generator); silence-only shortcut test
Everything lives in libri_exp\\ (clips, features, model, report).
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, soundfile as sf
from sv_audio import (ROOT, SR, WIN, load16, save16, rms_norm, trim_silence, to16k, train_windows, silence_only,
                      gate_pauses, random_tilt, add_noise, telephonize, pad_to, rng_for)
from scipy.signal import fftconvolve
from sv_heads import eer, platt, sigmoid

N_SPK, N_UTT = 100, 8                  # 800 real + 800 XTTS clones (XTTS ~25-40 min on your GPU)
N_TEST, N_CAL = 20, 10                 # speaker-disjoint: 70 fit / 10 calibrate / 20 test
PHONE_ONLY = True                      # every clip, real and fake, only as its phone-line (8 kHz G.711) copy
# noise / room settings tried one after another (same recipe for real AND fake, no pitch/speed change)
CONFIGS = {"A": dict(snr=(5, 30),  reverb=False),
           "B": dict(snr=(0, 20),  reverb=False),
           "C": dict(snr=(0, 20),  reverb=True),
           "D": dict(snr=(-5, 15), reverb=True)}
EXP = ROOT / "libri_exp"
MANIFEST, XDIR, KDIR, FEATS = EXP / "manifest.json", EXP / "xtts", EXP / "knnvc", EXP / "features.npz"
REPORT, MODEL = EXP / "report.json", EXP / "deepfake_libri_xtts.pt"

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else default

def find_libri():
    if arg("--libri"): return Path(arg("--libri"))
    for c in (Path(r"C:\Users\saksh\vg_demo\LibriSpeech\train-clean-100"), ROOT / "LibriSpeech" / "train-clean-100"):
        if c.exists(): return c
    for base in (Path(r"C:\Users\saksh\vg_demo"), ROOT):
        hits = list(base.glob("**/train-clean-100"))
        if hits: return hits[0]
    sys.exit('LibriSpeech not found - run:  python -u libri_generalization.py --libri "path\\to\\train-clean-100"')

def pretty(t):
    t = t.strip().lower()
    return (t[:1].upper() + t[1:]).replace(" i ", " I ") + "."

# ------------------------------------------------------------------ 1. select
def select():
    if MANIFEST.exists(): return json.loads(MANIFEST.read_text())
    libri = find_libri(); print(f"LibriSpeech: {libri}")
    rng = np.random.default_rng(0)
    spks = sorted((p for p in libri.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: p.name)
    rng.shuffle(spks)
    chosen = []
    for sp in spks:
        chapters = sorted(p for p in sp.iterdir() if p.is_dir())
        utts = []
        for ch in chapters:
            for tf in ch.glob("*.trans.txt"):
                for line in tf.read_text().splitlines():
                    uid, text = line.split(" ", 1)
                    f = ch / f"{uid}.flac"
                    if f.exists(): utts.append((ch.name, uid, str(f), text))
        dur = {u[1]: sf.info(u[2]).duration for u in utts}
        ok = [u for u in utts if 4 <= dur[u[1]] <= 12 and len(u[3]) <= 220]
        if len(ok) < N_UTT + 3: continue
        ref_ch = chapters[0].name if len(chapters) > 1 else None
        test_pool = [u for u in ok if u[0] != ref_ch] if ref_ch else ok[: len(ok) // 2]
        ref_pool = [u for u in utts if u[0] == ref_ch] if ref_ch else [u for u in utts if u not in test_pool]
        if len(test_pool) < N_UTT or not ref_pool: continue
        pick = [test_pool[i] for i in sorted(rng.choice(len(test_pool), N_UTT, replace=False))]
        refs, tot = [], 0.0
        for u in ref_pool:                                   # ~60 s of reference from the other chapter
            if tot >= 60: break
            refs.append(u[2]); tot += dur[u[1]]
        chosen.append({"spk": sp.name, "refs": refs, "ref_sec": round(tot, 1),
                       "utts": [{"uid": u[1], "path": u[2], "text": pretty(u[3])} for u in pick]})
        print(f"\r  selected {len(chosen)}/{N_SPK} speakers", end="", flush=True)
        if len(chosen) == N_SPK: break
    print()
    for i, c in enumerate(chosen):
        c["split"] = "test" if i >= len(chosen) - N_TEST else ("cal" if i >= len(chosen) - N_TEST - N_CAL else "fit")
    EXP.mkdir(parents=True, exist_ok=True); MANIFEST.write_text(json.dumps(chosen, indent=1))
    return chosen

# ------------------------------------------------------------------ audio loaders that avoid torchcodec
def patch_torchaudio():
    import torch, torchaudio
    def _load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, *a, **k):
        w, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        if frame_offset: w = w[frame_offset:]
        if num_frames and num_frames > 0: w = w[:num_frames]
        return torch.from_numpy(np.ascontiguousarray(w.T if channels_first else w)), sr
    torchaudio.load = _load

# ------------------------------------------------------------------ 2. XTTS clones
def make_xtts(man):
    todo = [(c, u) for c in man for u in c["utts"] if not (XDIR / f"{u['uid']}.wav").exists()]
    print(f"XTTS: {len(todo)} clones to make")
    if not todo: return
    import torch
    from tts_compat import TTS                             # your file that makes coqui-tts work on torch 2.9
    patch_torchaudio()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
    srm = getattr(tts.synthesizer, "output_sample_rate", 24000) or 24000
    XDIR.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for i, (c, u) in enumerate(todo, 1):
        try:
            wav = tts.tts(text=u["text"], speaker_wav=c["refs"][:3], language="en")
        except Exception as e:
            print(f"  FAILED {u['uid']}: {e}"); continue
        save16(XDIR / f"{u['uid']}.wav", rms_norm(trim_silence(to16k(np.asarray(wav, "float32"), srm))))
        left = (time.time() - t0) / i * (len(todo) - i) / 60
        print(f"[{i:4d}/{len(todo)}] {u['uid']}   ~{left:5.1f} min left")

# ------------------------------------------------------------------ 3. kNN-VC fakes (test speakers only)
def knn_pairs(man):
    test = [c for c in man if c["split"] == "test"]
    for i, c in enumerate(test):                            # target c, source = next test speaker's sentences
        src = test[(i + 1) % len(test)]
        for u in src["utts"]:
            yield c, u, KDIR / f"{c['spk']}__{u['uid']}.wav"

def make_knnvc(man):
    todo = [(c, u, out) for c, u, out in knn_pairs(man) if not out.exists()]
    print(f"kNN-VC: {len(todo)} fakes to make")
    if not todo: return
    import torch
    patch_torchaudio()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
    KDIR.mkdir(parents=True, exist_ok=True); cur = matching = None
    for i, (c, u, out) in enumerate(todo, 1):
        if c["spk"] != cur:
            matching = knn.get_matching_set(c["refs"], vad_trigger_level=0); cur = c["spk"]
        w = knn.match(knn.get_features(u["path"]), matching, topk=4).detach().cpu().numpy().astype("float32")
        save16(out, rms_norm(trim_silence(w)))
        print(f"[{i:3d}/{len(todo)}] speaker {c['spk']} voice <- {u['uid']}")

# ------------------------------------------------------------------ 4. features (same recipe for every class)
def room(w, rng):
    """random room echo: exponentially decaying noise tail, RT60 0.2-0.8 s, random wet/dry mix"""
    rt = rng.uniform(0.2, 0.8); n = int(rt * SR)
    h = rng.standard_normal(n).astype("float32") * np.exp(-6.9 * np.arange(n) / n).astype("float32")
    h[0] = 1.0; h /= np.sqrt((h ** 2).sum())
    wet = fftconvolve(w, h)[: len(w)].astype("float32")
    mix = rng.uniform(0.3, 1.0)
    return rms_norm(mix * wet + (1 - mix) * w)

def views(w, key, cfg):
    """gate pauses -> [room echo] -> mic tilt -> noise at cfg SNR -> phone line. Same for real and fake."""
    rng = rng_for(key)
    c = pad_to(rms_norm(gate_pauses(rms_norm(w))), WIN)
    x = room(c, rng) if cfg["reverb"] else c
    x = add_noise(rms_norm(random_tilt(x, rng)), rng, snr_db=cfg["snr"])
    return c, telephonize(x)

def features(man, name):
    cfg = CONFIGS[name]; FEATS = EXP / f"features_{name}.npz"
    if FEATS.exists(): return dict(np.load(FEATS))
    from sv_models import Encoder
    enc = Encoder()
    items = []                                              # (path, label 0/1, kind, speaker, split)
    for c in man:
        for u in c["utts"]:
            items.append((u["path"], 0, "real", c["spk"], c["split"]))
            if (XDIR / f"{u['uid']}.wav").exists(): items.append((str(XDIR / f"{u['uid']}.wav"), 1, "xtts", c["spk"], c["split"]))
    for c, u, out in knn_pairs(man):
        if out.exists(): items.append((str(out), 1, "knnvc", c["spk"], "test"))
    X, meta, S, smeta, buf, bmeta = [], [], [], [], [], []
    def flush():
        if buf: X.append(enc.numpy(buf, label="windows")); meta.extend(bmeta); buf.clear(); bmeta.clear()
    t0 = time.time()
    for n, (p, lab, kind, spk, split) in enumerate(items, 1):
        w = rms_norm(trim_silence(load16(p)))
        clean, x = views(w, f"libri|{name}|{kind}|{Path(p).stem}", cfg)
        for seg in train_windows(x):
            buf.append(seg); bmeta.append((lab, kind, spk, split, 1))
        s = silence_only(clean, x)
        if s is not None: S.append(s); smeta.append((lab, kind, spk, split))
        if len(buf) >= 240: flush()
        if n % 100 == 0: print(f"  {n}/{len(items)} clips processed ({time.time() - t0:.0f}s)")
    flush()
    XS = enc.numpy(S, label="silence-only")
    m = np.array(meta, dtype=object); sm = np.array(smeta, dtype=object)
    d = dict(X=np.concatenate(X), y=m[:, 0].astype("int8"), kind=m[:, 1].astype(str), spk=m[:, 2].astype(str),
             split=m[:, 3].astype(str), view=m[:, 4].astype("int8"),
             XS=XS, ys=sm[:, 0].astype("int8"), skind=sm[:, 1].astype(str), ssplit=sm[:, 3].astype(str))
    np.savez(FEATS, **d)
    print(f"[{name}] features: {d['X'].shape[0]} windows, {XS.shape[0]} silence snippets")
    return d

# ------------------------------------------------------------------ 5. train on XTTS, test on both
def evaluate(d, name="A"):
    from sv_models import fit_deepfake
    X, y, kind, split, view = d["X"], d["y"], d["kind"], d["split"], d["view"]
    fit = (split == "fit") & ((kind == "real") | (kind == "xtts"))
    cal = (split == "cal") & ((kind == "real") | (kind == "xtts"))
    print(f"training on {int(fit.sum())} windows from FIT speakers (real vs XTTS only) ...")
    m = fit_deepfake(X[fit], y[fit])
    a, b = platt(m.logits_np(X[cal]), y[cal])
    m.save(EXP / f"deepfake_libri_xtts_{name}.pt", trained_on=f"LibriSpeech real vs XTTS, setting {name}", calib_a=a, calib_b=b)
    te = split == "test"
    z = {k: m.logits_np(X[te & (kind == k)]) for k in ("real", "xtts", "knnvc")}
    def e(fake, ph=None):
        r, f = z["real"], z[fake]
        if ph is not None:
            r = r[view[te & (kind == "real")] == ph]; f = f[view[te & (kind == fake)] == ph]
        return eer(np.r_[r, f], np.r_[np.zeros(len(r)), np.ones(len(f))])[0] if len(f) else float("nan")
    p = {k: sigmoid(a * v + b) for k, v in z.items()}
    sil = float("nan")
    XS, ys, ssplit, skind = d["XS"], d["ys"], d["ssplit"], d["skind"]
    sf_, st_ = (ssplit == "fit") & (skind != "knnvc"), (ssplit == "test") & (skind != "knnvc")
    if len(set(ys[sf_].tolist())) == 2 and len(set(ys[st_].tolist())) == 2:
        sil = eer(fit_deepfake(XS[sf_], ys[sf_]).logits_np(XS[st_]), ys[st_])[0]
    rep = dict(speakers={s: int(len(set(d["spk"][split == s]))) for s in ("fit", "cal", "test")},
               eer_xtts_seen=e("xtts"), eer_knnvc_unseen=e("knnvc"),
               eer_xtts_phone=e("xtts", 1), eer_knnvc_phone=e("knnvc", 1),
               real_flagged=float((p["real"] > .5).mean()), xtts_caught=float((p["xtts"] > .5).mean()),
               knnvc_caught=float((p["knnvc"] > .5).mean()) if len(p["knnvc"]) else float("nan"),
               silence_only_eer=sil, layer_weights=[round(float(v), 3) for v in m.layer_weights()],
               calib=dict(a=a, b=b), setting=CONFIGS[name])
    return rep

def show(reps):
    REPORT.write_text(json.dumps(reps, indent=2))
    pc = lambda v: "  n/a" if v != v else f"{100 * v:5.1f}%"
    first = next(iter(reps.values()))
    print("\n" + "=" * 96)
    print(f"LIBRISPEECH, all audio phone-line G.711, trained on XTTS only | speakers {first['speakers']['fit']} fit / "
          f"{first['speakers']['cal']} cal / {first['speakers']['test']} test (unseen)")
    print("=" * 96)
    print("setting  noise (SNR)   room echo | silence-only | EER XTTS (seen) | EER kNN-VC (unseen) | kNN caught | real flagged")
    for n, r in reps.items():
        c = r["setting"]
        print(f"   {n}     {c['snr'][0]:>3} to {c['snr'][1]:>2} dB   {'yes' if c['reverb'] else 'no ':<3}      |   {pc(r['silence_only_eer'])}     |"
              f"     {pc(r['eer_xtts_seen'])}      |       {pc(r['eer_knnvc_unseen'])}        |   {pc(r['knnvc_caught'])}   |   {pc(r['real_flagged'])}")
    print("\nwant: silence-only near 50% (pauses carry no clue), then the lowest kNN-VC EER.")
    for n, r in reps.items():
        print(f"  layer weights {n}: " + " ".join(f"L{i}:{v:.2f}" for i, v in enumerate(r["layer_weights"])))
    print(f"saved {REPORT.relative_to(ROOT)}  (your app's models are untouched)")

if __name__ == "__main__":
    man = select()
    print(f"{len(man)} speakers: {sum(c['split'] == 'fit' for c in man)} fit / {sum(c['split'] == 'cal' for c in man)} cal / "
          f"{sum(c['split'] == 'test' for c in man)} test, {N_UTT} sentences each")
    if arg("--stage") != "eval":
        make_xtts(man); make_knnvc(man)
    names = arg("--configs", ",".join(CONFIGS)).split(",")
    reps = {}
    for n in names:
        print(f"\n----- setting {n}: noise {CONFIGS[n]['snr']} dB, room echo {CONFIGS[n]['reverb']} -----")
        reps[n] = evaluate(features(man, n), n)
    show(reps)
