"""Train the deepfake head on XTTS and kNN-VC together; test on unseen LibriSpeech speakers.

    python experiments/libri_two_generators.py [--eval]

Run experiments/libri_generalization.py first. Adds kNN-VC fakes for the training and calibration speakers
(voices swapped within each group, so no test audio reaches training), uses noise setting A and a
strict silence test (frames at least 100 ms from speech). Trains three heads (XTTS only, kNN-VC
only, both) and writes libri_exp/two_generators.json.
"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the sv_* modules
from sv_audio import SR, load16, save16, rms_norm, trim_silence, train_windows, frame_db
from sv_heads import eer, platt, sigmoid
import libri_generalization as LG

CFG = LG.CONFIGS["A"]
FEATS2 = LG.EXP / "features_2gen.npz"
REPORT2 = LG.EXP / "two_generators.json"

def pairs_all(man):
    """(target, source utterance, output path) for every group: fit, cal, test"""
    for grp in ("fit", "cal", "test"):
        g = [c for c in man if c["split"] == grp]
        for i, c in enumerate(g):
            for u in g[(i + 1) % len(g)]["utts"]:
                yield grp, c, u, LG.KDIR / f"{c['spk']}__{u['uid']}.wav"

def make_knnvc_all(man):
    todo = [(c, u, out) for grp, c, u, out in pairs_all(man) if not out.exists()]
    print(f"kNN-VC: {len(todo)} fakes still to make (all groups)")
    if not todo: return
    import torch
    LG.patch_torchaudio()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
    LG.KDIR.mkdir(parents=True, exist_ok=True); cur = matching = None; t0 = time.time()
    for i, (c, u, out) in enumerate(todo, 1):
        if c["spk"] != cur:
            matching = knn.get_matching_set(c["refs"], vad_trigger_level=0); cur = c["spk"]
        w = knn.match(knn.get_features(u["path"]), matching, topk=4).detach().cpu().numpy().astype("float32")
        save16(out, rms_norm(trim_silence(w)))
        if i % 20 == 0 or i == len(todo):
            print(f"  [{i:4d}/{len(todo)}]  ~{(time.time() - t0) / i * (len(todo) - i) / 60:4.1f} min left")

def strict_silence(clean, x, gap_frames=5, min_sec=0.15):
    """frames at least gap_frames*20 ms away from any speech frame of the (gated) clean signal"""
    n = min(len(clean), len(x)) // 320
    e = frame_db(clean[: n * 320])
    speech = e > max(e.max() - 35.0, np.percentile(e, 10) + 10.0)
    near = np.convolve(speech.astype(float), np.ones(2 * gap_frames + 1), "same") > 0
    idx = np.where(~near)[0]
    if len(idx) * 320 < min_sec * SR: return None
    s = np.concatenate([x[i * 320: (i + 1) * 320] for i in idx])
    return np.tile(s, int(np.ceil(SR / len(s))))[:SR].astype("float32")

def features(man):
    if FEATS2.exists(): return dict(np.load(FEATS2))
    from sv_models import Encoder
    enc = Encoder()
    items = []
    for c in man:
        for u in c["utts"]:
            items.append((u["path"], 0, "real", c["spk"], c["split"]))
            xp = LG.XDIR / f"{u['uid']}.wav"
            if xp.exists(): items.append((str(xp), 1, "xtts", c["spk"], c["split"]))
    for grp, c, u, out in pairs_all(man):
        if out.exists(): items.append((str(out), 1, "knnvc", c["spk"], grp))
    X, meta, S, smeta, buf, bmeta = [], [], [], [], [], []
    def flush():
        if buf: X.append(enc.numpy(buf, label="windows")); meta.extend(bmeta); buf.clear(); bmeta.clear()
    t0 = time.time()
    for n, (p, lab, kind, spk, split) in enumerate(items, 1):
        clean, x = LG.views(rms_norm(trim_silence(load16(p))), f"2gen|{kind}|{Path(p).stem}", CFG)
        for seg in train_windows(x): buf.append(seg); bmeta.append((lab, kind, spk, split))
        s = strict_silence(clean, x)
        if s is not None: S.append(s); smeta.append((lab, kind, spk, split))
        if len(buf) >= 240: flush()
        if n % 200 == 0: print(f"  {n}/{len(items)} clips ({time.time() - t0:.0f}s)")
    flush()
    XS = enc.numpy(S, label="strict silence")
    m = np.array(meta, dtype=object)
    sm = np.array(smeta, dtype=object).reshape(-1, 4)          # may be empty if clips have no long pauses
    d = dict(X=np.concatenate(X), y=m[:, 0].astype("int8"), kind=m[:, 1].astype(str), spk=m[:, 2].astype(str),
             split=m[:, 3].astype(str), XS=XS, ys=sm[:, 0].astype("int8"), skind=sm[:, 1].astype(str), ssplit=sm[:, 3].astype(str))
    np.savez(FEATS2, **d)
    counts = {k: int((d["kind"] == k).sum()) for k in ("real", "xtts", "knnvc")}
    print(f"features: {counts} windows, {len(XS)} strict-silence snippets")
    return d

def run(d):
    from sv_models import fit_deepfake
    X, y, kind, split = d["X"], d["y"], d["kind"], d["split"]
    te = split == "test"
    zr = None
    rows = {}
    for name, fakes in (("XTTS only", {"xtts"}), ("kNN-VC only", {"knnvc"}), ("BOTH", {"xtts", "knnvc"})):
        use = lambda grp: (split == grp) & ((kind == "real") | np.isin(kind, list(fakes)))
        print(f"\ntraining '{name}' on {int(use('fit').sum())} windows ...")
        m = fit_deepfake(X[use("fit")], y[use("fit")])
        a, b = platt(m.logits_np(X[use("cal")]), y[use("cal")])
        m.save(LG.EXP / f"deepfake_2gen_{name.split()[0].lower().replace('-', '')}.pt", trained_on=name, calib_a=a, calib_b=b)
        z = {k: m.logits_np(X[te & (kind == k)]) for k in ("real", "xtts", "knnvc")}
        def e(*ks):
            f = np.concatenate([z[k] for k in ks])
            return eer(np.r_[z["real"], f], np.r_[np.zeros(len(z["real"])), np.ones(len(f))])[0]
        p = {k: sigmoid(a * v + b) for k, v in z.items()}
        rows[name] = dict(eer_xtts=e("xtts"), eer_knnvc=e("knnvc"), eer_both=e("xtts", "knnvc"),
                          xtts_caught=float((p["xtts"] > .5).mean()), knnvc_caught=float((p["knnvc"] > .5).mean()),
                          real_flagged=float((p["real"] > .5).mean()),
                          layer_weights=[round(float(v), 3) for v in m.layer_weights()])
    XS, ys, ssplit = d["XS"], d["ys"], d["ssplit"]
    sil = float("nan")
    if len(set(ys[ssplit == "fit"].tolist())) == 2 and len(set(ys[ssplit == "test"].tolist())) == 2:
        sil = eer(fit_deepfake(XS[ssplit == "fit"], ys[ssplit == "fit"]).logits_np(XS[ssplit == "test"]), ys[ssplit == "test"])[0]
    REPORT2.write_text(json.dumps({"rows": rows, "strict_silence_eer": sil, "setting": CFG}, indent=2))
    pc = lambda v: "  n/a" if v != v else f"{100 * v:5.1f}%"
    print("\n" + "=" * 104)
    print("TWO GENERATORS - LibriSpeech, phone line, setting A (5-30 dB), tested on 20 UNSEEN speakers")
    print("=" * 104)
    print(f"{'trained on':<13}|{'EER vs XTTS':>12} {'EER vs kNN-VC':>14} {'EER vs both':>12} | {'XTTS caught':>11} {'kNN caught':>11} {'real flagged':>13}")
    for n, r in rows.items():
        print(f"{n:<13}|{pc(r['eer_xtts']):>12} {pc(r['eer_knnvc']):>14} {pc(r['eer_both']):>12} | "
              f"{pc(r['xtts_caught']):>11} {pc(r['knnvc_caught']):>11} {pc(r['real_flagged']):>13}")
    print(f"\nstrict silence-only EER (frames >= 100 ms from any word; want ~50%): {pc(sil)}"
          f"   [{len(XS)} clips had long enough pauses]")
    print("read it: 'XTTS only' vs kNN-VC = unseen generator; 'kNN-VC only' vs XTTS = unseen the other way;")
    print("         'BOTH' = what the app should use if it holds up on both columns.")
    for n, r in rows.items(): print(f"  layer weights {n:<11}: " + " ".join(f"L{i}:{v:.2f}" for i, v in enumerate(r["layer_weights"])))
    print(f"saved {REPORT2.name} and three models in libri_exp\\")

if __name__ == "__main__":
    man = LG.select()
    print(f"{len(man)} speakers: " + ", ".join(f"{g} {sum(c['split'] == g for c in man)}" for g in ("fit", "cal", "test")))
    if "--eval" not in sys.argv: make_knnvc_all(man)
    run(features(man))
