"""Audio helpers shared by the pipeline and the app (numpy, scipy and soundfile only)."""
import audioop, subprocess, zlib
from fractions import Fraction
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, lfilter

# ------------------------------------------------------------------ paths
ROOT      = Path(__file__).resolve().parent
RAW       = ROOT / "raw"                 # your phone recordings, flat
SENT      = ROOT / "sentences"           # s01.txt .. s06.txt
DATA      = ROOT / "data"
REAL_DIR  = DATA / "real"                # s01_001.wav ...      (step 1)
REF_DIR   = DATA / "ref"                 # s01.wav ...          (step 1)
XTTS_DIR  = DATA / "xtts"                # xtts__s01_001.wav    (step 2)
FEATS     = DATA / "features.npz"        #                      (step 3)
MODELS    = ROOT / "models"
SPK_CKPT  = MODELS / "spk_head.pt"       # your trained speaker head (copied in)
DF_DIR    = MODELS / "deepfake"          # fold_s01.pt ... all.pt calib.json (step 4)
VOICEPRINTS = MODELS / "voiceprints.npz" #                      (step 5)
REPORTS   = ROOT / "reports"             # json + mapping files

# ------------------------------------------------------------------ team
#            id     name        first last   <- global file numbers of their 30 sentences
SPEAKERS = [("s01", "Saksham",    1,  30),
            ("s02", "Asmit",     31,  60),
            ("s03", "Vijay",     61,  90),
            ("s04", "Nikhil",    91, 120),
            ("s05", "Yogya",    121, 150),
            ("s06", "Shreyas",  151, 181)]
NAME = {sid: name for sid, name, *_ in SPEAKERS}
ENROL_CLIPS = range(1, 11)     # sentences 001-010 build the voiceprint
TEST_CLIPS  = range(11, 31)    # sentences 011-030 are only ever used for testing / demo calls

# ------------------------------------------------------------------ constants
SR  = 16000
WIN = 3 * SR                   # every model looks at exactly 3 s
TRAIN_HOP = 3 * SR // 2        # 1.5 s hop when cutting training windows
MAXWIN = 4
SNR_DB = (5.0, 30.0)

# ------------------------------------------------------------------ basic I/O
def to16k(w, sr):
    w = np.asarray(w, dtype="float32")
    if sr == SR: return w
    g = gcd(SR, int(sr))
    return resample_poly(w, SR // g, int(sr) // g).astype("float32")

class StreamResampler:
    """Live mic -> 16 kHz without seams: resamples a rolling buffer and only hands out samples far
    enough from the buffer edges, so 0.5 s pieces join exactly as if the whole recording was converted at once."""
    GUARD = 800                                   # hold back the last 50 ms (edge of the filter) until more arrives
    def __init__(self, sr):
        g = gcd(SR, int(sr)); self.sr, self.up, self.down = int(sr), SR // g, int(sr) // g
        self.raw, self.r0, self.out = np.zeros(0, "float32"), 0, 0
    def push(self, x):
        self.raw = np.concatenate([self.raw, np.asarray(x, "float32")])
        if self.up == self.down == 1:
            y, o0 = self.raw, self.r0
        else:
            y, o0 = resample_poly(self.raw, self.up, self.down).astype("float32"), self.r0 * self.up // self.down
        end = o0 + len(y) - (0 if self.up == self.down else self.GUARD)
        new = y[self.out - o0: end - o0] if end > self.out else np.zeros(0, "float32")
        self.out = max(self.out, end)
        keep = ((self.out - SR) * self.down // self.up) // self.down * self.down     # keep ~1 s of context
        if keep > self.r0:
            self.raw = self.raw[keep - self.r0:]; self.r0 = keep
        return new

def load16(path):
    """Any audio file -> mono float32 at 16 kHz, DC removed. WAV/FLAC via soundfile, else ffmpeg."""
    path = Path(path)
    try:
        w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        tmp = ROOT / "tmp_decode.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                            "-ac", "1", "-ar", str(SR), str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"cannot read {path.name}: {r.stderr.strip()}")
        w, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    w = to16k(w.mean(1), sr)
    return (w - w.mean()).astype("float32")

def save16(path, w):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(w, -1, 1), SR, subtype="PCM_16")

def duration(path):
    try:
        i = sf.info(str(path)); return i.frames / i.samplerate
    except Exception:
        return len(load16(path)) / SR

# ------------------------------------------------------------------ cleaning
def frame_db(w, frame=320):
    n = len(w) // frame
    if n == 0: return np.full(1, -120.0)
    return 20 * np.log10(np.sqrt((w[: n * frame].reshape(n, frame) ** 2).mean(1)) + 1e-9)

def trim_silence(w, keep=0.15, rel_db=35.0):
    """Cut leading/trailing silence, keep a short pad. A frame is 'live' if it is within rel_db of the
    loudest frame AND 10 dB above this clip's own noise floor (so a fan or hiss doesn't count as speech)."""
    e = frame_db(w)
    live = np.where(e > max(e.max() - rel_db, np.percentile(e, 10) + 10))[0]
    if len(live) == 0: return w
    pad = int(keep * SR)
    return w[max(0, live[0] * 320 - pad): min(len(w), (live[-1] + 1) * 320 + pad)]

def rms_norm(w, target=0.05):
    w = np.asarray(w, dtype="float32")
    w = w * (target / (np.sqrt((w ** 2).mean()) + 1e-9))
    p = np.abs(w).max() if len(w) else 0.0
    return (w * (0.99 / p) if p > 0.99 else w).astype("float32")

def pad_to(w, n):
    if len(w) >= n: return w
    a = (n - len(w)) // 2
    return np.pad(w, (a, n - len(w) - a)).astype("float32")

# ------------------------------------------------------------------ channel + noise
def telephonize(w):
    """Phone line: 16 kHz -> 8 kHz -> G.711 mu-law -> 16 kHz (kills everything above 4 kHz)."""
    w8 = resample_poly(w, 1, 2)
    pcm = (np.clip(w8, -1, 1) * 32767).astype("<i2").tobytes()
    w8 = np.frombuffer(audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2), dtype="<i2").astype("float32") / 32768.0
    out = resample_poly(w8, 2, 1).astype("float32")
    return out[: len(w)] if len(out) >= len(w) else np.pad(out, (0, len(w) - len(out)))

def add_noise(w, rng, snr_db=SNR_DB):
    """Coloured background noise at a random SNR (white ... rumbly)."""
    snr = rng.uniform(*snr_db)
    a = rng.uniform(0.0, 0.95)
    n = lfilter([1 - a], [1, -a], rng.standard_normal(len(w))).astype("float32")
    k = np.sqrt((np.mean(w ** 2) + 1e-9) / ((np.mean(n ** 2) + 1e-9) * 10 ** (snr / 10)))
    return (w + k * n).astype("float32")

def rng_for(key):
    return np.random.default_rng(zlib.crc32(str(key).encode()))

def gate_pauses(w, fade=160):
    """Silence the gaps between words (keep 40 ms around speech, 10 ms fades). Applied to real AND fake
    before the noise goes in, so every pause ends up as the same kind of random noise: the room tone of
    a phone recording vs the clean pauses of XTTS can no longer tell the classes apart."""
    e = frame_db(w)
    speech = (e > max(e.max() - 35.0, np.percentile(e, 10) + 10.0)).astype("float32")
    speech = (np.convolve(speech, np.ones(5, "float32"), "same") > 0).astype("float32")
    g = np.repeat(speech, 320)
    g = np.pad(g, (0, max(0, len(w) - len(g))), constant_values=float(g[-1]) if len(g) else 1.0)[: len(w)]
    g = np.convolve(g, np.ones(fade, "float32") / fade, "same")
    return (w * g).astype("float32")

def random_tilt(w, rng):
    """Random spectral tilt (mic colouring): y[n] = x[n] - c*x[n-1], c in [-0.4, 0.6]."""
    c = rng.uniform(-0.4, 0.6)
    return lfilter([1.0, -c], [1.0], w).astype("float32")

SPEEDS = (0.9, 0.95, 1.0, 1.05, 1.1)

def speed_perturb(w, factor):
    """Play faster/slower: factor 1.1 = 10% faster and ~1.7 semitones higher. Changes pitch the way a raised,
    excited or tired voice does (Kaldi-style speed perturbation)."""
    if abs(factor - 1.0) < 1e-6: return w
    fr = Fraction(1.0 / factor).limit_denominator(50)
    return resample_poly(w, fr.numerator, fr.denominator).astype("float32")

def augment_views(w, key):
    """Two training copies of one clip, SAME recipe for real and fake:
    random speed/pitch (0.9-1.1x) -> gate the pauses -> random mic tilt -> pad to 3 s -> background noise
    (5-30 dB SNR); view 0 = that, view 1 = that through a phone line.
    Returns (view, clean, processed); 'clean' marks where speech is (for the silence-only test)."""
    g = gate_pauses(rms_norm(w))
    for v in (0, 1):
        rng = rng_for(f"{key}|{v}")
        c = pad_to(rms_norm(speed_perturb(g, SPEEDS[rng.integers(len(SPEEDS))])), WIN)
        x = add_noise(rms_norm(random_tilt(c, rng)), rng)
        yield v, c, (telephonize(x) if v == 1 else x)

# ------------------------------------------------------------------ windows
def train_windows(x):
    """3 s windows with 1.5 s hop, at most MAXWIN per clip."""
    L = len(x)
    if L <= WIN: return [pad_to(x, WIN)[:WIN]]
    st = list(range(0, L - WIN + 1, TRAIN_HOP))
    if st[-1] != L - WIN: st.append(L - WIN)
    if len(st) > MAXWIN: st = [st[int(round(i))] for i in np.linspace(0, len(st) - 1, MAXWIN)]
    return [x[s: s + WIN] for s in st]

def silence_only(clean, proc, rel_db=30.0, min_sec=0.3):
    """Only the frames where nobody speaks, stitched and tiled to 1 s (shortcut test)."""
    n = min(len(clean), len(proc)) // 320
    e = frame_db(clean[: n * 320])
    idx = np.where(e < max(e.max() - rel_db, np.percentile(e, 10) + 6))[0]
    if len(idx) * 320 < min_sec * SR: return None
    s = np.concatenate([proc[i * 320: (i + 1) * 320] for i in idx])
    return np.tile(s, int(np.ceil(SR / len(s))))[:SR].astype("float32")

def speech_fraction(w):
    """Energy VAD on RAW audio: share of 20 ms frames that are speech.
    Speech = at least 10 dB above this window's own noise floor and above -60 dBFS."""
    e = frame_db(w)
    if e.max() < -55: return 0.0
    floor = np.percentile(e, 10)
    return float(np.mean((e > floor + 10) & (e > -60)))

# ------------------------------------------------------------------ sentences + clips
def load_sentences(sid):
    f = SENT / f"{sid}.txt"
    if not f.exists(): return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cid, lang, text = [x.strip() for x in line.split("|", 2)]
            out[cid] = (lang, text)
    return out

def real_clip(cid):  return REAL_DIR / f"{cid}.wav"
def xtts_clip(cid):  return XTTS_DIR / f"xtts__{cid}.wav"

def stitch(paths, gap=0.35, seed=0):
    """Join clips into one 'call' with short gaps of faint room noise (never digital silence)."""
    rng = np.random.default_rng(seed)
    parts, marks, t = [], [], 0.0
    for p in paths:
        w = rms_norm(load16(p))
        g = (rng.standard_normal(int(gap * SR)) * 0.002).astype("float32")
        parts += [w, g]; marks.append((t, t + len(w) / SR)); t += (len(w) + len(g)) / SR
    return (np.concatenate(parts) if parts else np.zeros(SR, "float32")), marks
