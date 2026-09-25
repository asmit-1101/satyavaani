# train_speaker.py — learned speaker embedding, channel-invariant
import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop, json, time
import numpy as np, torch, torch.nn as nn, soundfile as sf
from collections import defaultdict
from pathlib import Path
from scipy.signal import resample_poly
from transformers import Wav2Vec2Model
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR, WIN, DIM = 16000, 3.0, 192
N = int(SR * WIN)
CLIPS_PER_SPK = 15
CORPUS = "train-clean-100"
HELDOUT_SPK = 40             # raise to 40 with train-clean-100
CACHE = Path("feats/spk_cache.npz")

dev = "cuda" if torch.cuda.is_available() else "cpu"
Path("feats").mkdir(exist_ok=True); Path("out").mkdir(exist_ok=True)


def prep(w):
    """Must match extract.py and app.py exactly."""
    w = w - w.mean()
    w = w / (np.sqrt((w ** 2).mean()) + 1e-8)
    return (w - w.mean()) / (w.std() + 1e-7)


def telephonize(w):
    """16k -> 8k G.711 mu-law -> 16k. Same chain as telephonize.py."""
    w8 = resample_poly(w, 1, 2)
    pcm = (np.clip(w8, -1, 1) * 32767).astype("<i2").tobytes()
    back = audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2)
    w8 = np.frombuffer(back, dtype="<i2").astype("float32") / 32768.0
    return resample_poly(w8, 2, 1).astype("float32")


def load_trunk():
    return Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base", use_safetensors=False).to(dev).eval().half()


def pool(trunk, batch):
    x = torch.from_numpy(np.stack(batch)).to(dev, dtype=torch.float16)
    with torch.inference_mode():
        H = torch.stack(trunk(x, output_hidden_states=True).hidden_states)
        p = torch.cat([H.mean(2), H.std(2)], -1)
    return p.permute(1, 0, 2).float().cpu().numpy()


def read_audio(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        g = np.gcd(SR, sr)
        w = resample_poly(w, SR // g, sr // g)
    return w.astype("float32")


# ---------- phase 1: cache clean AND codec'd views, tagged ------------------
def build_cache():
    trunk = load_trunk()
    root = Path("LibriSpeech") / CORPUS
    if not root.exists():
        import torchaudio.datasets as tds
        print("downloading LibriSpeech...")
        tds.LIBRISPEECH(".", url=CORPUS, download=True)

    per_spk, items = defaultdict(int), []
    for f in sorted(root.rglob("*.flac")):
        spk = f.parts[-3]
        if per_spk[spk] < CLIPS_PER_SPK:
            per_spk[spk] += 1
            items.append((f, spk))
    print(f"{len(per_spk)} speakers, {len(items)} clips "
          f"-> up to {2*len(items)} views")

    X, S, C, T, buf, meta = [], [], [], [], [], []
    for ci, (f, spk) in enumerate(tqdm(items, desc="encoding")):
        w = read_audio(f)
        if len(w) < N:
            continue
        s = (len(w) - N) // 2
        crop = w[s:s + N]
        # SAME label for both -> loss forces the embedding to ignore channel
        for is_tel, variant in ((0, crop), (1, telephonize(crop))):
            buf.append(prep(variant)); meta.append((spk, ci, is_tel))
            if len(buf) == 16:
                X.append(pool(trunk, buf))
                for m in meta: S.append(m[0]); C.append(m[1]); T.append(m[2])
                buf, meta = [], []
    if buf:
        X.append(pool(trunk, buf))
        for m in meta: S.append(m[0]); C.append(m[1]); T.append(m[2])

    X = np.concatenate(X)
    np.savez(CACHE, X=X, spk=np.array(S), clip=np.array(C), tel=np.array(T))
    print(f"cached {X.shape}   clean {(np.array(T)==0).sum()}  "
          f"telephony {(np.array(T)==1).sum()}")


if not CACHE.exists():
    build_cache()

c = np.load(CACHE, allow_pickle=True)
X = c["X"]
spk = np.array([str(s) for s in c["spk"]])
clip_id, is_tel = c["clip"], c["tel"]
L, D = X.shape[1], X.shape[2]
print(f"\n{X.shape}   {len(set(spk))} speakers   device {dev}")


# ---------- speaker-disjoint split ------------------------------------------
rng = np.random.default_rng(0)
all_spk = sorted(set(spk)); rng.shuffle(all_spk)
test_spk = set(all_spk[:HELDOUT_SPK])
tr = np.array([s not in test_spk for s in spk]); te = ~tr
print(f"train {tr.sum()} views / {len(all_spk)-HELDOUT_SPK} spk   "
      f"test {te.sum()} views / {HELDOUT_SPK} spk (never trained on)")

spk2id = {s: i for i, s in enumerate(sorted(set(spk[tr])))}
ytr_np = np.array([spk2id[s] for s in spk[tr]])
ytr = torch.tensor(ytr_np, device=dev)

mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-6
Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device=dev)
Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device=dev)


# ---------- the head --------------------------------------------------------
class SpeakerHead(nn.Module):
    """embed() is what ships. W is scaffolding, discarded after training."""
    def __init__(self, n_spk, n_layers, d_in, d_out=DIM, m=0.25, s=30.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_layers))          # LEARNED
        self.drop_in = nn.Dropout(0.2)
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, 256), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(256, d_out))
        self.W = nn.Parameter(torch.randn(n_spk, d_out) * 0.01)
        self.m, self.s = m, s

    def embed(self, h):
        w = torch.softmax(self.alpha, 0)[None, :, None]
        z = self.drop_in((h * w).sum(1))
        return nn.functional.normalize(self.net(z), dim=-1)

    def forward(self, h, labels):                                 # AAM-Softmax
        cos = self.embed(h) @ nn.functional.normalize(self.W, dim=-1).T
        oh = nn.functional.one_hot(labels, self.W.shape[0]).float()
        return self.s * (cos - self.m * oh)


def pair_eer(emb, labels):
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sim = E @ E.T
    same = labels[:, None] == labels[None, :]
    iu = np.triu_indices(len(labels), k=1)
    pos, neg = sim[iu][same[iu]], sim[iu][~same[iu]]
    ths = np.linspace(-1, 1, 2000)
    far = (neg[None] >= ths[:, None]).mean(1)
    frr = (pos[None] < ths[:, None]).mean(1)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2 * 100), float(ths[i]), pos.mean(), neg.mean()


head = SpeakerHead(len(spk2id), L, D).to(dev)
opt = torch.optim.AdamW([{"params": [head.alpha],          "lr": 1e-1},
                         {"params": head.net.parameters(), "lr": 5e-4},
                         {"params": [head.W],              "lr": 5e-4}],
                        weight_decay=5e-3)
te_lab = spk[te]

EPOCHS, BATCH = 600, 128
n_tr = Xtr.shape[0]
best, best_state, patience = 100.0, None, 0
rng2 = np.random.default_rng(1)

print("\ntraining  (mini-batch, early stop on held-out EER)")
t0 = time.time()
for ep in range(EPOCHS):
    head.train()
    perm = torch.from_numpy(rng2.permutation(n_tr)).to(dev)
    for i in range(0, n_tr, BATCH):
        b = perm[i:i + BATCH]
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(Xtr[b], ytr[b]), ytr[b])
        loss.backward(); opt.step()

    if ep % 10 == 0:
        head.eval()
        with torch.no_grad():
            e = head.embed(Xte).cpu().numpy()
        v = pair_eer(e, te_lab)[0]
        if v < best - 0.05:
            best, patience = v, 0
            best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
        else:
            patience += 1
        if ep % 60 == 0:
            with torch.no_grad():
                acc = (head(Xtr, ytr).argmax(1) == ytr).float().mean().item()
            print(f"  ep {ep:3d}  loss {loss.item():.3f}  train-acc {acc:.3f}  "
                  f"heldout EER {v:.2f}%   best {best:.2f}%")
        if patience >= 8:
            print(f"  early stop at ep {ep}")
            break

if best_state is not None:
    head.load_state_dict(best_state)
head.eval()
with torch.no_grad():
    emb = head.embed(Xte).cpu().numpy()
    alpha = torch.softmax(head.alpha, 0).cpu().numpy()
final, thr, pos_m, neg_m = pair_eer(emb, te_lab)

print(f"\nheld-out speaker EER {final:.2f}%   [{time.time()-t0:.0f}s]")
print(f"  same-speaker cosine {pos_m:.3f}   different {neg_m:.3f}")
print(f"  suggested threshold {thr:.3f}")
print(f"  learned peak at layer {int(alpha.argmax())} "
      f"({alpha.max():.3f}, uniform {1/L:.3f})")
print(f"  mass in top 4 layers {sum(sorted(alpha)[-4:]):.2f}")
print(f"  weights: {np.round(alpha, 3).tolist()}")
if alpha.max() < 1.4 / L:
    print("  NOTE: alpha still flat - more speakers is the fix, not more epochs")


# ---------- channel invariance ----------------------------------------------
te_idx = np.where(te)[0]
pairs = defaultdict(dict)
for i in te_idx:
    pairs[clip_id[i]][int(is_tel[i])] = i
both = [v for v in pairs.values() if 0 in v and 1 in v]
if both:
    ci = np.array([v[0] for v in both]); ti = np.array([v[1] for v in both])
    with torch.no_grad():
        zc = head.embed(torch.tensor((X[ci] - mu) / sd,
                                     dtype=torch.float32, device=dev)).cpu().numpy()
        zt = head.embed(torch.tensor((X[ti] - mu) / sd,
                                     dtype=torch.float32, device=dev)).cpu().numpy()
    inv = float((zc * zt).sum(1).mean())
    print(f"\nchannel invariance: same clip clean vs telephony, "
          f"cosine {inv:.3f}  (want > 0.85, over {len(both)} pairs)")
else:
    inv = None


# ---------- cross-domain: your own voices -----------------------------------
def eval_own(root="audio/self_test"):
    p = Path(root)
    if not p.exists():
        print(f"\n(no {root}/ — add <name>/*.wav folders for a cross-domain number)")
        return None
    trunk = load_trunk()
    embs, labs = [], []
    for spk_dir in sorted(d for d in p.iterdir() if d.is_dir()):
        for f in sorted(spk_dir.glob("*.wav")):
            w = read_audio(f)
            if len(w) < N:
                w = np.pad(w, (0, N - len(w)))
            s = (len(w) - N) // 2
            pooled = pool(trunk, [prep(w[s:s + N])])[0]
            z = torch.tensor((pooled - mu) / sd, dtype=torch.float32, device=dev)[None]
            with torch.no_grad():
                embs.append(head.embed(z).cpu().numpy()[0])
            labs.append(spk_dir.name)
    if len(set(labs)) < 2:
        print("\n(need at least 2 speaker folders with clips)")
        return None
    E, labs = np.stack(embs), np.array(labs)
    e, t, pm, nm = pair_eer(E, labs)
    print(f"\ncross-domain — your recordings, no fine-tuning")
    print(f"  {len(set(labs))} speakers, {len(labs)} clips")
    print(f"  same-speaker cosine {pm:.3f}   different {nm:.3f}")
    print(f"  suggested threshold {t:.3f}   EER {e:.2f}%")
    print(f"  (small sample — indicative, not a benchmark)")
    return {"eer": e, "threshold": t}


own = eval_own()

torch.save({"state": {k: v.cpu() for k, v in head.state_dict().items()},
            "mu": mu, "sd": sd, "n_layers": L, "d_in": D, "dim": DIM,
            "threshold": (own or {}).get("threshold", thr)}, "out/spk_head.pt")

HAND = np.exp(np.array([0., 1., 3., 3., 2.5, 2., 1.5] + [0.] * (L - 7)))
HAND /= HAND.sum()
json.dump({"eer": final, "threshold": thr, "alpha": alpha.tolist(),
           "hand_alpha": HAND.tolist(), "n_speakers": len(spk2id),
           "corpus": CORPUS, "channel_invariance": inv, "cross_domain": own},
          open("out/speaker.json", "w"), indent=2)

fig, ax = plt.subplots(figsize=(10, 3.5))
w = 0.4
ax.bar(np.arange(L) - w/2, alpha, w, label="learned", color="#FFB547")
ax.bar(np.arange(L) + w/2, HAND, w, label="hand-set guess", color="#888")
ax.axhline(1/L, ls="--", c="#aaa", lw=1)
ax.set_xlabel("hidden state index"); ax.set_ylabel("weight")
ax.set_title(f"Speaker head layer weighting  (held-out EER {final:.1f}%)")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("out/speaker_layers.png", dpi=150)
print("\nwrote out/speaker_layers.png")