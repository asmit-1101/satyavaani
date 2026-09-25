"""Export the learned weights of every trained head, in one go.
    python export_layer_weights.py                      (models\deepfake, libri_exp, spk_head)
    python export_layer_weights.py models_backup        (also any backup folders you name)
Reads every .pt in models\\deepfake\\, libri_exp\\ and models\\spk_head.pt (nothing is changed or retrained).
Writes:
  reports\\layer_weights.csv   - one row per model: how much it relies on each wav2vec2 layer L0..L12 (sums to 1)
  reports\\layer_weights.json  - the same, plus per-seed values
  reports\\all_parameters\\<model>.npz - every weight matrix and bias of that head (numpy arrays)
"""
import csv, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"; PAR = OUT / "all_parameters"
FILES = sorted((ROOT / "models" / "deepfake").glob("*.pt")) + sorted((ROOT / "libri_exp").glob("*.pt")) \
        + [ROOT / "models" / "spk_head.pt"] + [f for d in sys.argv[1:] for f in sorted(Path(d).rglob("*.pt"))]

def softmax(a):
    a = np.asarray(a, "float64"); e = np.exp(a - a.max()); return e / e.sum()

def np_(v): return v.detach().cpu().float().numpy() if torch.is_tensor(v) else np.asarray(v)

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flatten(v, f"{prefix}{k}."))
        elif torch.is_tensor(v): out[f"{prefix}{k}"] = np_(v)
    return out

rows, full = [], {}
for p in FILES:
    if not p.exists(): continue
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    name = f"{p.parent.name}/{p.stem}"
    if name in full: name = f"{p.parent.parent.name}/{name}"
    if isinstance(ck, dict) and "states" in ck:                       # deepfake head (ensemble of seeds)
        seeds = [softmax(np_(st["alpha"])) for st in ck["states"]]
        kind = "deepfake"
        params = {f"seed{i}.{k}": np_(v) for i, st in enumerate(ck["states"]) for k, v in st.items()}
        params["mu"], params["sd"] = np_(ck["mu"]), np_(ck["sd"])
        meta = {k: v for k, v in ck.items() if k not in ("states", "mu", "sd") and not torch.is_tensor(v)}
    else:                                                             # speaker head (any checkpoint layout)
        params = flatten(ck if isinstance(ck, dict) else {"state": ck.state_dict()})
        a = [v for k, v in params.items() if "alpha" in k.lower() and v.ndim == 1] or \
            [v for k, v in params.items() if v.ndim == 1 and v.size == 13 and not k.endswith("bias")
             and k.split(".")[-1] not in ("mu", "sd", "mean", "std")]
        if not a: print(f"  {name}: no layer-weight vector found, skipped"); continue
        seeds, kind = [softmax(a[0])], "speaker"
        meta = {k: v for k, v in ck.items() if isinstance(v, (str, int, float))} if isinstance(ck, dict) else {}
    w = np.mean(seeds, 0)
    rows.append([name, kind] + [round(float(x), 4) for x in w])
    full[name] = {"kind": kind, "layer_weights": w.round(4).tolist(),
                  "per_seed": [s.round(4).tolist() for s in seeds], "meta": {k: str(v) for k, v in meta.items()}}
    PAR.mkdir(parents=True, exist_ok=True)
    np.savez(PAR / (name.replace("/", "__") + ".npz"), **params)
    top = np.argsort(w)[::-1][:3]
    print(f"{name:<34} {kind:<9} top: " + "  ".join(f"L{i} {w[i]:.2f}" for i in top))

OUT.mkdir(exist_ok=True)
with open(OUT / "layer_weights.csv", "w", newline="") as f:
    cw = csv.writer(f); cw.writerow(["model", "head"] + [f"L{i}" for i in range(13)]); cw.writerows(rows)
(OUT / "layer_weights.json").write_text(json.dumps(full, indent=2))
print(f"\n{len(rows)} models -> reports\\layer_weights.csv, reports\\layer_weights.json, reports\\all_parameters\\")
