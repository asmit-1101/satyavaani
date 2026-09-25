"""Build voiceprints three ways (clean mic, phone line, or the average) and compare them.

    python experiments/compare_voiceprints.py

Each teammate is enrolled from sentences 001-010 and tested on 011-030 heard the same way.
Reports genuine/impostor EER and how many XTTS clones pass. Writes reports/voiceprint_compare.json.
"""
import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the sv_* modules
from sv_audio import (SPEAKERS, NAME, ENROL_CLIPS, TEST_CLIPS, REPORTS, WIN, real_clip, xtts_clip,
                      load16, rms_norm, trim_silence, pad_to, train_windows, telephonize)
from sv_models import Encoder, load_speaker_head
from sv_heads import eer, voiceprint

# voiceprint built from        -> tested on (matched)
MODES = {"codec only":      (("phone",),          ("phone",)),
         "mic only":        (("clean",),          ("clean",)),
         "mic + codec avg": (("clean", "phone"),  ("clean", "phone"))}   # test = every clip both ways, pooled

def clip_embs(enc, spk, path):
    w = rms_norm(trim_silence(load16(path)))
    out = {}
    for ch, x in (("clean", w), ("phone", telephonize(w))):
        out[ch] = spk.embed(enc(train_windows(pad_to(x, WIN))).cpu().numpy())      # (windows, 192)
    return out

def main():
    enc, spk = Encoder(), load_speaker_head()
    team = [s for s, *_ in SPEAKERS]
    E = {}                                              # (kind, cid) -> {"clean": embs, "phone": embs}
    for s in team:
        for k in range(1, 31):
            cid = f"{s}_{k:03d}"
            if real_clip(cid).exists(): E[("real", cid)] = clip_embs(enc, spk, real_clip(cid))
            if k in TEST_CLIPS and xtts_clip(cid).exists(): E[("xtts", cid)] = clip_embs(enc, spk, xtts_clip(cid))
        print(f"\r  embedded {s}", end="", flush=True)
    print()

    results = {}
    for mode, (chans, test_chans) in MODES.items():
        vps = {}
        for s in team:
            embs = [E[("real", f"{s}_{k:03d}")][ch] for k in ENROL_CLIPS if ("real", f"{s}_{k:03d}") in E for ch in chans]
            if embs: vps[s] = voiceprint(np.concatenate(embs))
        tch = " + ".join(test_chans)
        if True:
            gen, imp, clo = [], [], []
            for (kind, cid), ch in E.items():
                if int(cid[-3:]) not in TEST_CLIPS: continue
                owner = cid[:3]
                for tc in test_chans:
                    v = voiceprint(ch[tc])
                    for s, vp in vps.items():
                        c = float(vp @ v)
                        if kind == "real": (gen if owner == s else imp).append(c)
                        elif owner == s: clo.append(c)
            gen, imp, clo = map(np.array, (gen, imp, clo))
            e, thr = eer(np.r_[gen, imp], np.r_[np.ones(len(gen)), np.zeros(len(imp))])
            results[(mode, tch)] = dict(eer=e, threshold=thr, genuine=float(gen.mean()), impostor=float(imp.mean()),
                                        clone=float(clo.mean()) if len(clo) else None,
                                        gap=float(gen.mean() - imp.mean()),
                                        clone_pass=float((clo >= thr).mean()) if len(clo) else None,
                                        n_genuine=len(gen), n_impostor=len(imp))

    pc = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}%"
    print("\n" + "=" * 100)
    print("VOICEPRINT COMPARISON, matched: enrol 001-010 and test 011-030 on the same kind of audio")
    print("=" * 100)
    print(f"{'voiceprint built from':<18} {'tested on':<13} | {'EER':>6} | {'threshold':>9} | {'genuine':>7} {'other':>6} {'clone':>6} "
          f"| {'gap':>5} | {'clones pass':>11}")
    for (mode, tch), r in results.items():
        print(f"{mode:<18} {tch:<13} | {pc(r['eer'])} | {r['threshold']:9.3f} | {r['genuine']:7.2f} {r['impostor']:6.2f} "
              f"{(r['clone'] or 0):6.2f} | {r['gap']:5.2f} | {pc(r['clone_pass']):>11}")
    key = {m: (m, " + ".join(MODES[m][1])) for m in MODES}
    best = min(MODES, key=lambda m: results[key[m]]["eer"])
    print(f"\nlowest EER (each voiceprint on its matching audio): {best}  ({100 * results[key[best]]['eer']:.1f}%)")
    n = results[key["codec only"]]
    print(f"note: {n['n_genuine']} genuine and {n['n_impostor']} impostor trials - differences under ~2 points are noise.")
    print("gap = genuine similarity minus other-teammate similarity: bigger = voices better separated.")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "voiceprint_compare.json").write_text(json.dumps({f"{m} | test {t}": r for (m, t), r in results.items()}, indent=2))
    print("saved reports\\voiceprint_compare.json")

if __name__ == "__main__":
    main()
