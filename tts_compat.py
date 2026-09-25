"""Lets coqui-tts run on torch 2.9+ without torchcodec: audio reading goes through soundfile.
Order matters: transformers loads first (sees no torchcodec, uses its normal path),
then coqui's torchcodec check is satisfied by a stub."""
import sys, types, importlib.machinery, importlib.metadata
import numpy as np, soundfile as sf, torch

# 1. load the transformers pieces XTTS needs while torchcodec is honestly absent
from transformers import GenerationMixin, GPT2Config, GPT2PreTrainedModel, LogitsProcessorList  # noqa: F401
import transformers.audio_utils  # noqa: F401
import transformers.utils.import_utils as _iu

# 2. now satisfy coqui's "torchcodec required" check with a stub
def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, *a, **k):
    w, sr = sf.read(str(uri), dtype="float32", always_2d=True)          # (frames, channels)
    if frame_offset: w = w[frame_offset:]
    if num_frames and num_frames > 0: w = w[:num_frames]
    return torch.from_numpy(np.ascontiguousarray(w.T if channels_first else w)), sr

try:
    import torchcodec  # noqa: F401
except Exception:
    class _Unavailable:
        def __init__(self, *a, **k):
            raise RuntimeError("torchcodec is stubbed - audio is read with soundfile instead")
    for name in ("torchcodec", "torchcodec.decoders", "torchcodec.encoders"):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        m.__version__ = "0.8.0"
        sys.modules[name] = m
    sys.modules["torchcodec.decoders"].AudioDecoder = _Unavailable
    sys.modules["torchcodec.encoders"].AudioEncoder = _Unavailable
    sys.modules["torchcodec"].decoders = sys.modules["torchcodec.decoders"]
    sys.modules["torchcodec"].encoders = sys.modules["torchcodec.encoders"]
    if hasattr(_iu, "is_torchcodec_available"):
        _iu.is_torchcodec_available = lambda: True
    _real_version = importlib.metadata.version
    importlib.metadata.version = lambda name: "0.8.0" if name == "torchcodec" else _real_version(name)

import torchaudio
torchaudio.load = _sf_load

from TTS.api import TTS  # noqa: E402