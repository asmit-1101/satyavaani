"""Redraw docs\\chart_layers.png: which wav2vec2 layers each deepfake head relies on, per training set.
    python export_layer_weights.py      (first, if reports\\layer_weights.csv is missing)
    python make_layer_chart.py
Reads reports\\layer_weights.csv. Needs matplotlib (pip install matplotlib).
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
ROWS = [  # (model in layer_weights.csv, label on the chart)
    ("deepfake/all",                 "Our team: real vs XTTS"),
    ("libri_exp/deepfake_2gen_xtts",  "LibriSpeech: trained on XTTS only"),
    ("libri_exp/deepfake_2gen_knnvc", "LibriSpeech: trained on kNN-VC only"),
    ("libri_exp/deepfake_2gen_both",  "LibriSpeech: trained on BOTH"),
    ("models/spk_head",              "Speaker head (for comparison)"),
]

table = {r[0]: [float(x) for x in r[2:15]] for r in csv.reader(open(ROOT / "reports" / "layer_weights.csv")) if r[0] != "model"}
rows = [(lab, table[m]) for m, lab in ROWS if m in table]
missing = [m for m, _ in ROWS if m not in table]
if missing: print("not found in layer_weights.csv (skipped):", ", ".join(missing))
W = np.array([w for _, w in rows])

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})
fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(rows) + 1.3), dpi=220)
fig.patch.set_facecolor(SURF)
ax.imshow(W, cmap=cmap, vmin=0, vmax=max(0.5, W.max()), aspect="auto")
for i in range(W.shape[0]):
    for j in range(13):
        v = W[i, j]
        if v >= 0.05:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.28 else INK, fontweight="bold" if v >= 0.2 else "normal")
ax.set_xticks(range(13)); ax.set_xticklabels([f"L{i}" for i in range(13)], color=INK2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for lab, _ in rows], color=INK)
ax.set_xticks(np.arange(-.5, 13), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
ax.grid(which="minor", color=SURF, linewidth=2); ax.tick_params(which="both", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlabel("wav2vec2 layer (L0 = right after the CNN, L12 = top)\nEach row sums to 1; values under 0.05 not printed",
              color=INK2, fontsize=8)
if any(lab.startswith("Speaker") for lab, _ in rows):
    ax.axhline(len(rows) - 1.5, color=INK2, linewidth=1)
fig.tight_layout()
out = ROOT / "docs" / "chart_layers.png"; out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=SURF)
print(f"saved {out}")
for lab, w in rows:
    top = np.argsort(w)[::-1][:3]
    print(f"  {lab:<38} " + "  ".join(f"L{i} {w[i]:.2f}" for i in top))
