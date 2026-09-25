"""Step 3: wav2vec2 features for every real clip and its clone.

    python pipeline/step3_features.py

Real and fake go through exactly the same processing (sv_audio.augment_views): speed/pitch
variation, pauses gated and filled with background noise, random mic tilt, and a phone-line copy
of every clip, cut into 3 s windows. Pause-only snippets are saved too, for the shortcut test.
Writes data/features.npz.
"""
import time
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the sv_* modules
from sv_audio import FEATS, REAL_DIR, XTTS_DIR, load16, augment_views, train_windows, silence_only
from sv_models import Encoder

def main():
    real = {p.stem: p for p in REAL_DIR.glob("*.wav")}
    fake = {p.stem[6:]: p for p in XTTS_DIR.glob("xtts__*.wav")}
    ids = sorted(set(real) & set(fake))
    print(f"{len(ids)} matched pairs (a real clip + the XTTS clone of the same sentence)")
    if not ids: raise SystemExit("Nothing to do - run pipeline/step1_prep.py and pipeline/step2_clone.py first.")

    W, y, spk, clip, view, S, sy, sspk = [], [], [], [], [], [], [], []
    for label, src in ((0, real), (1, fake)):
        for cid in ids:
            for v, clean, x in augment_views(load16(src[cid]), f"{label}|{cid}"):
                for seg in train_windows(x):
                    W.append(seg); y.append(label); spk.append(cid[:3]); clip.append(cid); view.append(v)
                s = silence_only(clean, x)
                if s is not None:
                    S.append(s); sy.append(label); sspk.append(cid[:3])
    print(f"{len(W)} speech windows (3 s)  |  {len(S)} silence-only snippets")

    t0 = time.time()
    enc = Encoder(); print(f"encoder on {enc.dev}")
    X = enc.numpy(W, label="speech windows")
    XS = enc.numpy(S, label="silence only  ")
    FEATS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FEATS, X=X, y=np.array(y, "int8"), spk=np.array(spk), clip=np.array(clip),
             view=np.array(view, "int8"), XS=XS, ys=np.array(sy, "int8"), spks=np.array(sspk))
    print(f"saved {FEATS.name}: X {X.shape}, silence {XS.shape}  in {time.time() - t0:.0f}s")
    print("Next: python pipeline/step4_deepfake.py")

if __name__ == "__main__":
    main()
