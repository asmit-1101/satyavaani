"""SATYAVAANI - the models.
  Encoder       frozen wav2vec2-base, one pass -> 13 layers -> mean+std -> (B, 13, 1536)
  DeepfakeModel learned softmax over the 13 layers -> small MLP -> logit   (ensemble of seeds)
  load_speaker_head  YOUR trained speaker head -> numpy rebuild (sv_heads.SpeakerNP)
"""
import os, json, warnings
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
warnings.filterwarnings("ignore")
from collections import OrderedDict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sv_audio import DF_DIR, SPK_CKPT

MODEL_ID = "facebook/wav2vec2-base"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================== encoder
class Encoder:
    def __init__(self, device=DEV):
        from transformers import Wav2Vec2Model
        try:
            m = Wav2Vec2Model.from_pretrained(MODEL_ID, use_safetensors=False)
        except Exception:
            m = Wav2Vec2Model.from_pretrained(MODEL_ID)
        self.dev, self.half = device, device == "cuda"
        self.m = m.to(device).eval()
        if self.half: self.m = self.m.half()

    @torch.no_grad()
    def __call__(self, wins):
        """list of 1-D float32 arrays (same length) -> torch (B, 13, 1536) float32 on device"""
        x = np.stack([(w - w.mean()) / np.sqrt(w.var() + 1e-7) for w in wins]).astype("float32")
        t = torch.from_numpy(x).to(self.dev)
        if self.half: t = t.half()
        h = torch.stack(self.m(t, output_hidden_states=True).hidden_states, 1).float()   # B,13,T,768
        return torch.cat([h.mean(2), h.std(2)], -1)

    def numpy(self, wins, batch=24, label=""):
        """many windows -> numpy float16 (N, 13, 1536), with a progress line"""
        if not wins: return np.zeros((0, 13, 1536), "float16")
        out = []
        for i in range(0, len(wins), batch):
            out.append(self(wins[i:i + batch]).cpu().numpy().astype("float16"))
            print(f"\r  {label} {min(i + batch, len(wins))}/{len(wins)}", end="", flush=True)
        print()
        return np.concatenate(out)

# =================================================================== deepfake head
class DeepfakeHead(nn.Module):
    """softmax weight per layer -> weighted sum -> Dropout-Linear(1536,64)-ReLU-Dropout-Linear(64,1)"""
    def __init__(self, L=13, D=1536, H=64):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(L))
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, H), nn.ReLU(), nn.Dropout(0.5), nn.Linear(H, 1))
    def weights(self):
        return torch.softmax(self.alpha, 0)
    def forward(self, x):
        return self.net((self.weights()[None, :, None] * x).sum(1)).squeeze(-1)

class DeepfakeModel:
    """An ensemble of heads sharing one standardisation (mu, sd)."""
    def __init__(self, heads, mu, sd, device=DEV):
        self.dev = device
        self.heads = [h.to(device).eval() for h in heads]
        self.mu = torch.as_tensor(np.asarray(mu), dtype=torch.float32, device=device)
        self.sd = torch.as_tensor(np.asarray(sd), dtype=torch.float32, device=device)

    @torch.no_grad()
    def logits(self, feats):
        if isinstance(feats, np.ndarray): feats = torch.from_numpy(feats.astype("float32")).to(self.dev)
        x = (feats - self.mu) / self.sd
        return torch.stack([h(x) for h in self.heads]).mean(0)

    def logits_np(self, X, chunk=512):
        if len(X) == 0: return np.zeros(0)
        return np.concatenate([self.logits(X[i:i + chunk]).cpu().numpy() for i in range(0, len(X), chunk)])

    def layer_weights(self):
        return np.mean([h.weights().detach().cpu().numpy() for h in self.heads], 0)

    def save(self, path, **meta):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"states": [h.state_dict() for h in self.heads], "mu": self.mu.cpu(), "sd": self.sd.cpu(),
                    "L": int(self.mu.shape[1]), "D": int(self.mu.shape[2]), **meta}, str(path))

    @classmethod
    def load(cls, path, device=DEV):
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        heads = []
        for st in ck["states"]:
            h = DeepfakeHead(ck["L"], ck["D"]); h.load_state_dict(st); heads.append(h)
        return cls(heads, ck["mu"].numpy(), ck["sd"].numpy(), device)

def fit_deepfake(X, y, seeds=(0, 1, 2), epochs=60, bs=128, device=DEV):
    X = X.astype("float32")
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-5
    Xt = torch.from_numpy((X - mu) / sd).to(device)
    yt = torch.from_numpy(np.asarray(y, "float32")).to(device)
    heads = []
    for seed in seeds:
        torch.manual_seed(seed)
        h = DeepfakeHead(X.shape[1], X.shape[2]).to(device)
        opt = torch.optim.AdamW([{"params": [h.alpha], "lr": 5e-2, "weight_decay": 0.0},
                                 {"params": h.net.parameters(), "lr": 1e-3, "weight_decay": 1e-2}])
        for _ in range(epochs):
            h.train()
            for b in torch.randperm(len(Xt), device=device).split(bs):
                loss = F.binary_cross_entropy_with_logits(h(Xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        heads.append(h.eval())
    return DeepfakeModel(heads, mu, sd, device)

def load_deepfake_bank(device=DEV):
    """{'all': model, 's01': fold model that never saw s01, ...} + calibration (a, b)"""
    bank = {}
    for p in sorted(Path(DF_DIR).glob("*.pt")):
        key = "all" if p.stem == "all" else p.stem.replace("fold_", "")
        bank[key] = DeepfakeModel.load(p, device)
    cal = json.loads((Path(DF_DIR) / "calib.json").read_text()) if (Path(DF_DIR) / "calib.json").exists() \
        else {"a": 1.0, "b": 0.0}
    return bank, cal

# =================================================================== speaker head (loader)
def load_speaker_head(path=SPK_CKPT, verbose=True):
    """torch.load your spk_head.pt, hand the tensors to the numpy rebuild in sv_heads.py"""
    from sv_heads import SpeakerNP, split_checkpoint
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - copy your old out\\spk_head.pt into models\\")
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    state, extras = split_checkpoint(ck)
    conv = lambda v: v.detach().cpu().float().numpy() if torch.is_tensor(v) else v
    return SpeakerNP(OrderedDict((k, conv(v)) for k, v in state.items()),
                     {k: conv(v) for k, v in extras.items()}, verbose=verbose)
