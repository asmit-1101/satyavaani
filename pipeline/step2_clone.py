"""Step 2: an XTTS-v2 clone of every recorded sentence, in the same person's voice.

    python pipeline/step2_clone.py

Voice sample: data/ref/sNN.wav. Text: sentences/sNN.txt. Writes data/xtts/xtts__sNN_0NN.wav.
Safe to stop and rerun; existing clones are skipped. Needs coqui-tts.
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import warnings; warnings.filterwarnings("ignore")
import time
from math import gcd
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for the sv_* modules
from sv_audio import SPEAKERS, REAL_DIR, REF_DIR, XTTS_DIR, to16k, trim_silence, rms_norm, save16, load_sentences

def patch_reference_loader():
    """XTTS loads the reference with torchaudio.load (needs torchcodec on new torchaudio) -> use soundfile."""
    try:
        import TTS.tts.models.xtts as X
    except Exception:
        return False
    if not hasattr(X, "load_audio"): return False
    def load_audio(path, sampling_rate):
        w, sr = sf.read(str(path), dtype="float32", always_2d=True)
        w = w.mean(1)
        if sr != sampling_rate:
            g = gcd(sampling_rate, sr); w = resample_poly(w, sampling_rate // g, sr // g)
        return torch.from_numpy(np.clip(w, -1, 1).astype("float32"))[None]
    X.load_audio = load_audio
    return True

def main():
    jobs = []
    for sid, name, *_ in SPEAKERS:
        ref = REF_DIR / f"{sid}.wav"
        if not ref.exists(): print(f"skip {sid}: no {ref.name} - run pipeline/step1_prep.py"); continue
        for cid, (lang, text) in load_sentences(sid).items():
            if (REAL_DIR / f"{cid}.wav").exists():
                jobs.append((cid, lang, text, str(ref)))
    todo = [j for j in jobs if not (XTTS_DIR / f"xtts__{j[0]}.wav").exists()]
    print(f"{len(jobs)} recorded sentences, {len(todo)} clones still to make")

    if todo:
        from tts_compat import TTS
        print("reference loader patched:", patch_reference_loader())
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu": print("WARNING: torch sees no GPU - this will be very slow. See README step 2.")
        t0 = time.time()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
        sr_model = getattr(tts.synthesizer, "output_sample_rate", 24000) or 24000
        print(f"XTTS loaded on {dev} in {time.time() - t0:.0f}s\n")
        t0 = time.time()
        for i, (cid, lang, text, ref) in enumerate(todo, 1):
            try:
                wav = tts.tts(text=text, speaker_wav=ref, language=lang)
            except Exception as e:
                print(f"  FAILED {cid}: {e}"); continue
            w = rms_norm(trim_silence(to16k(np.asarray(wav, dtype="float32"), sr_model)))
            save16(XTTS_DIR / f"xtts__{cid}.wav", w)
            left = (time.time() - t0) / i * (len(todo) - i) / 60
            print(f"[{i:3d}/{len(todo)}] {cid} {lang}  {len(w) / 16000:4.1f}s   ~{left:4.1f} min left")

    real = {p.stem for p in REAL_DIR.glob("*.wav")}
    fake = {p.stem[6:] for p in XTTS_DIR.glob("xtts__*.wav")}
    print(f"\nreal {len(real)} | clones {len(fake)} | matched pairs {len(real & fake)}")
    for sid, name, *_ in SPEAKERS:
        r = sum(x.startswith(sid) for x in real); f = sum(x.startswith(sid) for x in fake)
        print(f"  {sid} {name:<8} real {r:2d}  clones {f:2d}" + ("" if r == f else "   <-- mismatch"))
    if real - fake: print("no clone for:", sorted(real - fake))
    print("\nLISTEN to two or three (e.g. data\\xtts\\xtts__s01_011.wav, xtts__s04_027.wav in Hindi).")
    print("Next: python pipeline/step3_features.py")

if __name__ == "__main__":
    main()
