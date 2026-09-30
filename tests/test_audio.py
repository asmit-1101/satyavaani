"""Audio pipeline (sv_audio.py): resampling, the phone line, the speech detector."""
import numpy as np
from scipy.signal import resample_poly

from sv_audio import SR, StreamResampler, to16k, telephonize, speech_fraction, pad_to, rms_norm
from helpers import speechlike


def _band_energy(w, lo, hi):
    spec = np.abs(np.fft.rfft(w)) ** 2
    f = np.fft.rfftfreq(len(w), 1 / SR)
    return spec[(f >= lo) & (f < hi)].sum()


def test_to16k_changes_the_length_by_the_rate():
    w = np.random.default_rng(0).standard_normal(48000).astype("float32")     # 1 s at 48 kHz
    assert len(to16k(w, 48000)) == 16000
    assert to16k(w[:16000], SR) is not None and len(to16k(w[:16000], SR)) == 16000


def test_stream_resampler_has_no_seams():
    # the live mic arrives in 0.5 s pieces; joined output must equal converting the whole recording once
    sr = 48000
    x = np.random.default_rng(1).standard_normal(sr * 4).astype("float32") * 0.1
    rs = StreamResampler(sr)
    pieces = [rs.push(x[i:i + sr // 2]) for i in range(0, len(x), sr // 2)]
    streamed = np.concatenate(pieces)
    whole = resample_poly(x, 1, 3).astype("float32")
    n = len(streamed)
    assert n > 0.9 * len(whole)
    assert np.max(np.abs(streamed - whole[:n])) < 1e-4


def test_phone_line_removes_everything_above_4_khz():
    t = np.arange(SR) / SR
    w = (0.3 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)).astype("float32")
    tel = telephonize(w)
    assert len(tel) == len(w)
    kept_1k = _band_energy(tel, 900, 1100) / _band_energy(w, 900, 1100)
    kept_6k = _band_energy(tel, 5900, 6100) / _band_energy(w, 5900, 6100)
    assert kept_1k > 0.5          # speech band survives
    assert kept_6k < 0.01         # above 4 kHz is gone


def test_speech_detector():
    assert speech_fraction(np.zeros(3 * SR, "float32")) == 0.0
    assert speech_fraction(speechlike(3)) >= 0.4
    quiet_hiss = (1e-4 * np.random.default_rng(2).standard_normal(3 * SR)).astype("float32")
    assert speech_fraction(quiet_hiss) < 0.4


def test_pad_and_normalise():
    w = np.ones(100, "float32")
    p = pad_to(w, 300)
    assert len(p) == 300 and p.sum() == 100
    assert len(pad_to(np.ones(500, "float32"), 300)) == 500
    n = rms_norm(speechlike(1, level=0.5))
    assert abs(np.sqrt((n ** 2).mean()) - 0.05) < 0.01 and np.abs(n).max() <= 0.99 + 1e-6
