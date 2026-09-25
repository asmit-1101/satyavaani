"""Step 5: enrol the team with the speaker head and measure it.

    python pipeline/step5_speaker.py

Voiceprints come from sentences 001-010 through a phone line. Sentences 011-030 test genuine
callers, other teammates and XTTS clones. Writes models/voiceprints.npz and reports/speaker_test.json.
"""
import json
from collections import OrderedDict
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the sv_* modules
from sv_audio import (SPEAKERS, NAME, ENROL_CLIPS, TEST_CLIPS, REPORTS, WIN, real_clip, xtts_clip,
                      load16, rms_norm, trim_silence, pad_to, train_windows, telephonize, speed_perturb)
from sv_models import Encoder, load_speaker_head
from sv_heads import eer, save_voiceprints, voiceprint

def clip_vec(enc, spk, path, phone=False, speed=1.0):
    w = rms_norm(trim_silence(load16(path)))
    if speed != 1.0: w = speed_perturb(w, speed)
    if phone: w = telephonize(w)
    return voiceprint(spk.embed(enc(train_windows(pad_to(w, WIN))).cpu().numpy()))

def main():
    enc, spk = Encoder(), load_speaker_head()
    team = [s for s, *_ in SPEAKERS]
    E = {}
    for s in team:
        for k in range(1, 31):
            cid = f"{s}_{k:03d}"
            if real_clip(cid).exists():
                E[("real", cid, 0)] = clip_vec(enc, spk, real_clip(cid))
                E[("real", cid, 1)] = clip_vec(enc, spk, real_clip(cid), phone=True)
            if k in TEST_CLIPS and xtts_clip(cid).exists():
                E[("xtts", cid, 0)] = clip_vec(enc, spk, xtts_clip(cid))
                E[("xtts", cid, 1)] = clip_vec(enc, spk, xtts_clip(cid), phone=True)
        print(f"\r  embedded {s}", end="", flush=True)
    print()

    vps = OrderedDict()
    for s in team:
        # voiceprint from sentences 001-010, phone-line (codec) copies only
        enrol = [E[("real", f"{s}_{k:03d}", 1)] for k in ENROL_CLIPS if ("real", f"{s}_{k:03d}", 1) in E]
        if enrol: vps[s] = (NAME[s], voiceprint(enrol))
    if len(vps) < 2: raise SystemExit("Need real clips for at least 2 people - run pipeline/step1_prep.py first.")

    def trials(kind, phone):
        g, i = [], []                                  # genuine / other-person cosines
        for s in vps:
            for (kd, cid, ph), v in E.items():
                if kd == kind and ph == phone and int(cid[-3:]) in TEST_CLIPS:
                    (g if cid[:3] == s else i).append(float(vps[s][1] @ v))
        return np.array(g), np.array(i)

    gen, imp = trials("real", 0)
    gen_ph, imp_ph = trials("real", 1)
    clone, _ = trials("xtts", 0)
    clone_ph, _ = trials("xtts", 1)
    e, thr_team = eer(np.r_[gen, imp], np.r_[np.ones(len(gen)), np.zeros(len(imp))])
    e_ph, _ = eer(np.r_[gen_ph, imp_ph], np.r_[np.ones(len(gen_ph)), np.zeros(len(imp_ph))])
    # use the equal-error threshold measured on our team, unless it is implausible -> keep LibriSpeech's
    thr = thr_team if (0.05 < thr_team < 0.90 and e < 0.30) else spk.threshold
    rate = lambda a, t=thr: float((a >= t).mean()) if len(a) else float("nan")

    print(f"\nspeaker EER on the team: {100 * e:.1f}%  (phone line {100 * e_ph:.1f}%)")
    print(f"mean cosine  genuine {gen.mean():.2f} | other teammate {imp.mean():.2f} | XTTS clone {clone.mean():.2f}")
    print(f"threshold: LibriSpeech {spk.threshold:.3f}, our team {thr_team:.3f}  -> the app uses {thr:.3f}")
    print(f"  at {thr:.3f}: genuine accepted {100 * rate(gen):.0f}%, other teammates rejected {100 * (1 - rate(imp)):.0f}%, "
          f"XTTS clones PASS the voice check {100 * rate(clone):.0f}%  (phone line: {100 * rate(clone_ph):.0f}%)")
    print("\nper person          genuine  clone   clones passing")
    per = {}
    for s in vps:
        gs = [float(vps[s][1] @ v) for (kd, cid, ph), v in E.items() if kd == "real" and ph == 0 and cid[:3] == s and int(cid[-3:]) in TEST_CLIPS]
        cs = [float(vps[s][1] @ v) for (kd, cid, ph), v in E.items() if kd == "xtts" and ph == 0 and cid[:3] == s]
        per[s] = dict(name=NAME[s], genuine_cos=float(np.mean(gs)) if gs else None,
                      clone_cos=float(np.mean(cs)) if cs else None, clone_pass=rate(np.array(cs)))
        print(f"  {s} {NAME[s]:<8}      {per[s]['genuine_cos'] or 0:.2f}    {per[s]['clone_cos'] or 0:.2f}     {100 * per[s]['clone_pass']:.0f}%")

    if e > 0.30:
        print("\nWarning: the speaker head barely separates the teammates (EER > 30%). The feature pipeline")
        print("   probably differs from the one it was trained with (see ref_code/train_speaker.py).")
    elif e > 0.15:
        print("\nNote: speaker EER is weaker than on LibriSpeech (different mics, accents, phone). Expected cross-domain.")

    save_voiceprints(vps)
    rep = dict(arch=spk.arch, eer=e, eer_phone=e_ph, threshold_librispeech=spk.threshold, threshold_team=thr_team,
               threshold_used=thr, genuine_accept=rate(gen), impostor_reject=1 - rate(imp),
               clone_pass_rate=rate(clone), clone_pass_rate_phone=rate(clone_ph),
               genuine_accept_phone=rate(gen_ph), mean_cos=dict(genuine=float(gen.mean()), impostor=float(imp.mean()),
               clone=float(clone.mean()) if len(clone) else None), per_speaker=per,
               layer_weights=[round(float(v), 4) for v in spk.layer_weights()])
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "speaker_test.json").write_text(json.dumps(rep, indent=2))
    print("\nsaved models\\voiceprints.npz and reports\\speaker_test.json")
    print("Next: python pipeline/measure_latency.py   then   python app.py")

if __name__ == "__main__":
    main()
