"""Torch-free parts: speaker-head forward pass, voiceprints, EER and Platt calibration."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from sv_audio import VOICEPRINTS

EMB_DIM = 192
FEAT_WIDTHS = (768, 1536)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
def logit(p):   p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

# =================================================================== speaker head
def _leaf(k): return k.split(".")[-1].lower()

class SpeakerNP:
    """Your trained speaker head, rebuilt from its tensors (as numpy arrays).
    Reads: layer weights (alpha), optional standardisation (mu/sd), then the chain of
    LayerNorm / BatchNorm / Linear layers in the order they were registered, stopping at the
    192-d embedding (the AAM-softmax classifier after it is ignored)."""
    STAT_NAMES = ("mu", "mean", "feat_mu", "sd", "std", "feat_sd", "sigma")

    def __init__(self, state, extras=None, verbose=True):
        extras = extras or {}
        sd = OrderedDict((k, np.asarray(v, dtype="float32")) for k, v in state.items())
        ex_arr = {k: np.asarray(v, dtype="float32") for k, v in extras.items()
                  if hasattr(v, "shape") and np.asarray(v).dtype.kind in "fiu"}
        pool = OrderedDict(sd); pool.update(ex_arr)

        # 1. layer weights
        ak = next((k for k, v in pool.items() if "alpha" in k.lower() and v.ndim == 1), None) or \
             next((k for k, v in pool.items() if v.ndim == 1 and v.size in (13, 25)
                   and not k.endswith("bias") and _leaf(k) not in self.STAT_NAMES), None)
        if ak is None:
            raise ValueError("speaker checkpoint: no layer-weight vector (alpha) found")
        self.alpha = pool[ak].astype("float32")
        L = self.alpha.size

        # 2. standardisation
        def pick(names):
            for src in (ex_arr, sd):
                for k, v in src.items():
                    if _leaf(k) in names: return v
            return None
        mu, sg = pick(("mu", "mean", "feat_mu")), pick(("sd", "std", "feat_sd", "sigma"))
        self.pre = self.post = None
        if mu is not None and sg is not None and mu.size == sg.size:
            if mu.size % L == 0 and mu.size // L in FEAT_WIDTHS:
                self.pre = (mu.reshape(1, L, -1), sg.reshape(1, L, -1))
            else:
                self.post = (mu.reshape(1, -1), sg.reshape(1, -1))

        # 3. group the remaining params by module, keep registration order
        groups = OrderedDict()
        for k, v in sd.items():
            if k == ak or _leaf(k) in self.STAT_NAMES or _leaf(k) == "num_batches_tracked": continue
            prefix, _, leaf = k.rpartition(".")
            groups.setdefault(prefix, {})[leaf.lower()] = v

        # input width
        D = self.pre[0].shape[-1] if self.pre is not None else None
        if D is None:
            for g in groups.values():
                w = g.get("weight")
                if w is not None and w.shape[-1] in FEAT_WIDTHS: D = w.shape[-1]; break
        if D is None: raise ValueError("speaker checkpoint: cannot tell the input width (768 or 1536)")
        self.D = D

        # 4. walk the chain
        raw, desc, cur, self.pre_ln = [], [f"alpha({L})"], D, None
        for prefix, g in groups.items():
            w = g.get("weight")
            if w is None: continue
            b = g.get("bias")
            if w.ndim == 2 and w.shape == (L, D) and b is not None and b.shape == (L, D):
                self.pre_ln = (w, b); desc.insert(1, f"LayerNorm({L}x{D})"); continue
            if w.ndim == 1 and w.size == cur:
                if "running_mean" in g:
                    raw.append(("bn", w, b if b is not None else np.zeros(cur, "float32"), g["running_mean"], g["running_var"]))
                    desc.append(f"BatchNorm({cur})")
                else:
                    raw.append(("ln", w, b if b is not None else np.zeros(cur, "float32"))); desc.append(f"LayerNorm({cur})")
                continue
            if w.ndim == 2 and w.shape[1] == cur and cur != EMB_DIM:
                raw.append(("lin", w, b)); desc.append(f"Linear({cur}->{w.shape[0]})"); cur = w.shape[0]
        if not any(op[0] == "lin" for op in raw):
            raise ValueError("speaker checkpoint: no Linear layers found on the feature path")

        # 5. ReLU between blocks: after a Linear followed by a Linear, or after a norm that sits after a Linear
        ops, lin_idx = [], [i for i, op in enumerate(raw) if op[0] == "lin"]
        for i, op in enumerate(raw):
            ops.append(op)
            later_lin = any(j > i for j in lin_idx)
            if op[0] == "lin" and i + 1 < len(raw) and raw[i + 1][0] == "lin":
                ops.append(("relu",))
            elif op[0] in ("ln", "bn") and i > 0 and raw[i - 1][0] == "lin" and later_lin:
                ops.append(("relu",))
        self.ops = ops
        self.emb_dim = cur
        thr = None
        for k, v in extras.items():
            if "thr" in k.lower() and np.ndim(v) == 0:
                try: thr = float(v)
                except Exception: pass
        self.threshold = thr if thr is not None else 0.319
        chain = " -> ".join({"lin": "Linear", "ln": "LayerNorm", "bn": "BatchNorm", "relu": "ReLU"}[o[0]]
                            + (f"({o[1].shape[1]}->{o[1].shape[0]})" if o[0] == "lin" else "") for o in ops)
        self.arch = (f"alpha({L}) {'+ per-layer standardise ' if self.pre is not None else ''}"
                     f"-> weighted sum -> {chain}  =>  {self.emb_dim}-d")
        if verbose:
            print(f"speaker head: {self.arch} voiceprint, threshold {self.threshold:.3f}")

    def embed(self, feats):
        """feats (B, L, >=D) numpy -> (B, emb) L2-normalised"""
        x = np.asarray(feats, dtype="float32")[..., : self.D]
        if self.pre is not None: x = (x - self.pre[0]) / self.pre[1]
        if self.pre_ln is not None:
            m = x.mean(axis=(1, 2), keepdims=True); v = x.var(axis=(1, 2), keepdims=True)
            x = (x - m) / np.sqrt(v + 1e-5) * self.pre_ln[0] + self.pre_ln[1]
        w = np.exp(self.alpha - self.alpha.max()); w /= w.sum()
        z = (w[None, :, None] * x).sum(1)
        if self.post is not None: z = (z - self.post[0]) / self.post[1]
        for op in self.ops:
            if op[0] == "lin":
                z = z @ op[1].T + (op[2] if op[2] is not None else 0)
            elif op[0] == "ln":
                z = (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + 1e-5) * op[1] + op[2]
            elif op[0] == "bn":
                z = (z - op[3]) / np.sqrt(op[4] + 1e-5) * op[1] + op[2]
            elif op[0] == "relu":
                z = np.maximum(z, 0)
        return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)

    def layer_weights(self):
        w = np.exp(self.alpha - self.alpha.max()); return w / w.sum()

def split_checkpoint(ck):
    """torch.load() result -> (state dict, extras). Works for {'state': {...}, 'thr': ..} or a bare state dict."""
    def is_tensor_like(v): return hasattr(v, "shape") and hasattr(v, "dtype")
    if hasattr(ck, "state_dict") and callable(ck.state_dict):
        return OrderedDict(ck.state_dict()), {}
    if not isinstance(ck, dict): raise TypeError(f"unexpected checkpoint type {type(ck)}")
    nested = {k: v for k, v in ck.items() if isinstance(v, dict) and v and all(is_tensor_like(t) for t in v.values())}
    if nested:
        key = max(nested, key=lambda k: sum(int(np.prod(t.shape)) for t in nested[k].values()))
        return OrderedDict(nested[key]), {k: v for k, v in ck.items() if k != key}
    return (OrderedDict((k, v) for k, v in ck.items() if is_tensor_like(v)),
            {k: v for k, v in ck.items() if not is_tensor_like(v)})

# =================================================================== voiceprints
def load_voiceprints(path=VOICEPRINTS):
    if not Path(path).exists(): return OrderedDict()
    d = np.load(str(path))
    return OrderedDict((str(i), (str(n), e.astype("float32"))) for i, n, e in zip(d["ids"], d["names"], d["embs"]))

def save_voiceprints(vps, path=VOICEPRINTS):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ids = list(vps)
    embs = np.stack([vps[i][1] for i in ids]).astype("float32") if ids else np.zeros((0, EMB_DIM), "float32")
    np.savez(str(path), ids=np.array(ids), names=np.array([vps[i][0] for i in ids]), embs=embs)

def voiceprint(embs):
    """many window embeddings -> one L2-normalised voiceprint"""
    m = np.asarray(embs, "float32").mean(0)
    return m / (np.linalg.norm(m) + 1e-9)

# =================================================================== metrics
def eer(score, label):
    """Equal error rate (higher score = the positive class). Returns (eer, threshold)."""
    s, y = np.asarray(score, float), np.asarray(label, int)
    P, N = (y == 1).sum(), (y == 0).sum()
    if P == 0 or N == 0: return float("nan"), 0.0
    o = np.argsort(s); s, y = s[o], y[o]
    frr = np.concatenate([[0], np.cumsum(y == 1)]) / P
    far = 1 - np.concatenate([[0], np.cumsum(y == 0)]) / N
    k = int(np.argmin(np.abs(frr - far)))
    lo = s[k - 1] if k > 0 else s[0] - 1
    hi = s[k] if k < len(s) else s[-1] + 1
    return float((frr[k] + far[k]) / 2), float((lo + hi) / 2)

def platt(z, y, iters=100, ridge=1.0):
    """Fit p = sigmoid(a*z + b) on out-of-fold logits (Platt scaling with smoothed targets).
    Numerically safe: logits are scaled to unit spread first, every Newton step is backtracked
    until the loss goes down, and a small ridge keeps a finite even when the classes separate."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    s = z.std() + 1e-9
    x = z / s
    npos, nneg = y.sum(), len(y) - y.sum()
    t = np.where(y > 0.5, (npos + 1) / (npos + 2), 1 / (nneg + 2))
    def loss(a, b):
        u = a * x + b
        return float(np.sum(np.logaddexp(0, u) - t * u) + 0.5 * ridge * a * a)
    a, b = 1.0, 0.0
    L = loss(a, b)
    for _ in range(iters):
        p = sigmoid(a * x + b)
        g = np.array([np.sum((p - t) * x) + ridge * a, np.sum(p - t)])
        w = p * (1 - p)
        H = np.array([[np.sum(w * x * x) + ridge, np.sum(w * x)], [np.sum(w * x), np.sum(w) + 1e-6]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = 1e-3 * g
        lr, improved = 1.0, False
        while lr > 1e-10:
            na, nb = a - lr * step[0], b - lr * step[1]
            nL = loss(na, nb)
            if np.isfinite(nL) and nL <= L:
                improved = True; break
            lr *= 0.5
        if not improved: break
        done = abs(L - nL) < 1e-10
        a, b, L = na, nb, nL
        if done: break
    return float(a / s), float(b)
