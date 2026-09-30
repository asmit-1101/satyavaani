"""Helpers for the tests. The tests never load the trained models or the team's recordings:
the model-free parts are tested directly, and the live engine gets small stand-in models."""
from collections import OrderedDict

import numpy as np

SR = 16000


def speechlike(seconds, seed=0, level=0.1):
    """Noise that switches on and off like syllables, loud enough to count as speech."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    env = 0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 3.0 * t))          # 3 Hz on/off, never fully silent
    return (level * env * rng.standard_normal(n)).astype("float32")


def unit(v):
    v = np.asarray(v, "float32")
    return v / np.linalg.norm(v)


def make_engine(p_synth_seq, emb_seq, voiceprints=None, thr=0.75):
    """An Engine whose deepfake head returns the given probabilities (one per scored window, the last one
    repeats) and whose speaker head returns the given embeddings. calib a=1, b=0 so logits map straight
    back to those probabilities."""
    from sv_engine import Engine
    from sv_heads import logit

    state = {"i": 0}

    def featurize(wins):
        return wins

    def deepfake_logit(feats, source):
        p = p_synth_seq[min(state["i"], len(p_synth_seq) - 1)]
        return float(logit(p))

    def speaker_embed(feats):
        e = emb_seq[min(state["i"], len(emb_seq) - 1)]
        state["i"] += 1
        return np.asarray([e] * len(feats), "float32")

    vps = OrderedDict(voiceprints or {})
    return Engine(featurize, deepfake_logit, speaker_embed, thr, {"a": 1.0, "b": 0.0}, vps, {})
