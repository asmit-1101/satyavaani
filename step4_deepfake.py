"""STEP 4 - train the deepfake head and test it honestly.
    python step4_deepfake.py
Leave-one-speaker-out: for each teammate, train on the other 5 (voices + their XTTS clones) and test on
this one - their voice AND sentences were never seen. That fold's head is saved and is the one the app
uses whenever this teammate's voice is in a demo call.
Also: silence-only test (pauses only - should be ~50%), probability calibration (Platt) from the
out-of-fold scores, and a final head trained on all 6 for new voices (upload / microphone).
Writes models\\deepfake\\fold_sNN.pt, all.pt, calib.json and reports\\deepfake_loso.json
"""
import json
import numpy as np
from sv_audio import FEATS, DF_DIR, REPORTS, NAME
from sv_models import fit_deepfake, DEV
from sv_heads import eer, platt, sigmoid

def pct(x): return "  n/a" if x != x else f"{100 * x:5.1f}%"

def main():
    d = np.load(FEATS)
    X, y, spk, view = d["X"], d["y"], d["spk"], d["view"]
    XS, ys, spks = d["XS"], d["ys"], d["spks"]
    speakers = sorted(set(spk.tolist()))
    print(f"{len(X)} windows, {len(speakers)} speakers, real {int((y == 0).sum())} / fake {int((y == 1).sum())}, on {DEV}\n")
    DF_DIR.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(exist_ok=True)

    oof, per, W, sil = np.zeros(len(X)), {}, [], []
    print("held out        EER    phone   clean | real flagged  fake caught | silence-only")
    for s in speakers:
        tr, te = spk != s, spk == s
        m = fit_deepfake(X[tr], y[tr])
        z = m.logits_np(X[te]); oof[te] = z
        e_all, _ = eer(z, y[te])
        e_tel = eer(z[view[te] == 1], y[te][view[te] == 1])[0]
        e_cln = eer(z[view[te] == 0], y[te][view[te] == 0])[0]
        p = sigmoid(z)
        fa, det = float((p[y[te] == 0] > 0.5).mean()), float((p[y[te] == 1] > 0.5).mean())
        e_sil = float("nan")
        str_, ste = spks != s, spks == s
        if len(set(ys[str_].tolist())) == 2 and len(set(ys[ste].tolist())) == 2:
            e_sil = eer(fit_deepfake(XS[str_], ys[str_]).logits_np(XS[ste]), ys[ste])[0]
        sil.append(e_sil)
        m.save(DF_DIR / f"fold_{s}.pt", held_out=s, eer=e_all)
        W.append(m.layer_weights())
        per[s] = dict(name=NAME.get(s, s), eer=e_all, eer_phone=e_tel, eer_clean=e_cln,
                      real_flagged=fa, fake_caught=det, silence_eer=e_sil)
        print(f"  {s} {NAME.get(s, s):<8} {pct(e_all)}  {pct(e_tel)}  {pct(e_cln)} |    {pct(fa)}       {pct(det)}   |    {pct(e_sil)}")

    a, b = platt(oof, y)
    json.dump({"a": a, "b": b, "note": "p_synthetic = sigmoid(a * logit + b), fitted on out-of-fold logits"},
              open(DF_DIR / "calib.json", "w"), indent=2)
    print("\ntraining the final head on all speakers (used for new voices: upload / microphone) ...")
    fit_deepfake(X, y).save(DF_DIR / "all.pt", held_out=None)

    e = np.array([per[s]["eer"] for s in speakers]); et = np.array([per[s]["eer_phone"] for s in speakers])
    silm = float(np.nanmean(sil)) if np.any(~np.isnan(sil)) else float("nan")
    Wm = np.mean(W, 0)
    pc = sigmoid(a * oof + b)
    rep = dict(speakers=speakers, per_speaker=per, eer_mean=float(e.mean()), eer_std=float(e.std()),
               eer_pooled=eer(oof, y)[0], eer_phone_mean=float(np.nanmean(et)), silence_eer_mean=silm,
               real_flagged=float((pc[y == 0] > 0.5).mean()), fake_caught=float((pc[y == 1] > 0.5).mean()),
               calib=dict(a=a, b=b), layer_weights=[round(float(v), 4) for v in Wm])
    (REPORTS / "deepfake_loso.json").write_text(json.dumps(rep, indent=2))

    print("\n" + "=" * 66 + "\nFOR THE SLIDES\n" + "=" * 66)
    print(f"  Deepfake EER, unseen speaker (6-fold LOSO) : {100 * e.mean():.1f}% +- {100 * e.std():.1f}%")
    print(f"  ... phone-line (G.711) copies only         : {100 * np.nanmean(et):.1f}%")
    print(f"  Silence-only EER (want ~50%)               : {100 * silm:.1f}%")
    print(f"  After calibration: real flagged {100 * rep['real_flagged']:.1f}%, clones caught {100 * rep['fake_caught']:.1f}%")
    print("\n  Layer weights the deepfake head learned (average over folds):")
    for i, v in enumerate(Wm): print(f"    L{i:<2} {v:.3f} {'#' * int(round(v * 150))}")
    if silm == silm and silm < 0.30:
        print("\n  !! Silence-only EER is low: the pauses alone still give real vs fake away,")
        print("     so part of the speech EER may be background, not voice. Say so honestly, or add more noise.")
    print(f"\nsaved models\\deepfake\\ (6 fold heads + all.pt + calib.json) and reports\\deepfake_loso.json")
    print("Next: python step5_speaker.py")

if __name__ == "__main__":
    main()
