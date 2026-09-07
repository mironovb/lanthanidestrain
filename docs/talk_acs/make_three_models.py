#!/usr/bin/env python3
"""three_models.png -- the three models and what each one reads.

Verified against the codebase (24 Aug 2026):
 * Column count comes from the collaborator's feature registry A2 =
   2,130 columns (collaborator_update/metrics_reproduction_20260818.md
   lines 102, 129, 325), independently recounted from his dataset.parquet:
   2,048 ECFP + 64 conditions + 10 RDKit + 4 metal + 4 complex plan.
   The fitted network carries weights for the 746 columns that vary; the
   other 1,384 (1,383 all-zero fingerprint bits and metal_ox, 3+ in every
   row) are constant and therefore inert -- a constant input only shifts a
   bias -- so the drawn width is the full 2,130-column registry.
 * The dense network trains on the same 746 columns (fcnn_diagnostic.py:158)
   with hidden_layer_sizes=(256, 128) (fcnn_diagnostic.py:110), so
   746 -> 256 -> 128 -> 1.
 * The 3D network concatenates the same 746 columns after pooling
   (snn.py:285; train.py:513); c15 configs record preset=baseline_2d.
 * Scores: CatBoost q60_rsm03_deep +0.2784 (C15_RESULTS.md), dense net
   16-seed +0.2206 (CONTROL_RESULTS.md), distance encoder c15_plw4 +0.2661
   and combined +0.3132 (c17_stack_new.csv).
Two populations, --population {legacy,collab}:
 * legacy  4,746 measurements ->   905 adjacent pairs, 162 extractants
 * collab  5,479 measurements -> 1,230 adjacent pairs, 177 extractants
   (the collaborator's Aug 2026 geometry repair).  Its scores come from
   automl/reports/collab_three_models.json, which retrains the fingerprint
   network on that population and refits the pair-level NNLS combination.
The molecular glyphs are the real Dy complex 2b190032c56e (24 heavy atoms),
PCA-projected, edges at the 3.5 A cutoff.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = "2b190032c56e"
CUT = 3.5

ap = argparse.ArgumentParser()
ap.add_argument("--population", default="legacy", choices=("legacy", "collab"))
POP = ap.parse_args().population

if POP == "legacy":
    S = dict(rows="4,746", pairs="905", trees="+0.28", fcnn="+0.22",
             enc="+0.27", comb="+0.31", out="three_models.png")
else:
    d = json.loads((HERE.parents[1] / "automl/reports"
                    / "collab_three_models.json").read_text())
    S = dict(rows=f"{d['n_rows']:,}", pairs=f"{d['n_pairs']:,}",
             trees=f"{d['trees']:+.2f}", fcnn=f"{d['fcnn']:+.2f}",
             enc=f"{d['enc3d']:+.2f}", comb=f"{d['combined']:+.2f}",
             out="three_models_collab.png")

INK, SUB, GREY = "#1a1a1a", "#666666", "#b9b9b9"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.dpi": 300})

# ------------------------------------------------------------ real complex
z = np.load(REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz")
bids = [str(b) for b in z["build_ids"]]
k = bids.index(BUILD)
a, b = int(z["node_ptr"][k]), int(z["node_ptr"][k + 1])
Z = z["atomic_numbers"][a:b]
xyz = z["coordinates"][a:b].astype(float)
ism = z["is_metal"][a:b].astype(bool)
heavy = Z > 1
xyz, ism = xyz[heavy], ism[heavy]
n = len(xyz)
c = xyz - xyz.mean(0)
_, _, vt = np.linalg.svd(c, full_matrices=False)
P = c @ vt[:2].T
P /= np.abs(P).max()
dm = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
edges = [(i, j) for i, j in combinations(range(n), 2) if dm[i, j] <= CUT]
metal = int(np.argmax(ism))

# ------------------------------------------------------------------ canvas
fig = plt.figure(figsize=(12.4, 6.2))
ax = fig.add_axes([0, 0.05, 1, 0.95])
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.set_aspect("equal")
ax.axis("off")


def molecule(cx, cy, s, color, lw=0.8, ms=14, mms=34):
    for i, j in edges:
        ax.plot(cx + s * P[[i, j], 0], cy + s * P[[i, j], 1],
                color=GREY, lw=lw, zorder=2, solid_capstyle="round")
    ax.scatter(cx + s * P[:, 0], cy + s * P[:, 1], s=ms, color=color,
               zorder=3, linewidths=0)
    ax.scatter(cx + s * P[metal, 0], cy + s * P[metal, 1], s=mms,
               color=INK, zorder=4, linewidths=0)


def table(cx, cy, w, h, rows=6, cols=5):
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white",
                           edgecolor=SUB, lw=1.0, zorder=2))
    ax.add_patch(Rectangle((x0, y0 + h * (rows - 1) / rows), w, h / rows,
                           facecolor=BLUE, edgecolor="none", zorder=2))
    for r in range(1, rows):
        ax.plot([x0, x0 + w], [y0 + h * r / rows] * 2, color=GREY, lw=0.6,
                zorder=3)
    for cc in range(1, cols):
        ax.plot([x0 + w * cc / cols] * 2, [y0, y0 + h], color=GREY, lw=0.6,
                zorder=3)


def tree(cx, cy, s):
    pts = {(0, 1): (cx, cy + s), (-1, 0): (cx - s * 0.75, cy),
           (1, 0): (cx + s * 0.75, cy),
           (-1.4, -1): (cx - s * 1.15, cy - s), (-0.6, -1): (cx - s * 0.35, cy - s),
           (0.6, -1): (cx + s * 0.35, cy - s), (1.4, -1): (cx + s * 1.15, cy - s)}
    for a_, b_ in (((0, 1), (-1, 0)), ((0, 1), (1, 0)),
                   ((-1, 0), (-1.4, -1)), ((-1, 0), (-0.6, -1)),
                   ((1, 0), (0.6, -1)), ((1, 0), (1.4, -1))):
        pa, pb = pts[a_], pts[b_]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=GREY, lw=1.1, zorder=2)
    for p in pts.values():
        ax.scatter(*p, s=22, color=BLUE, zorder=3, linewidths=0)


def net(cx, cy, w, h):
    """2,130 -> 256 -> 128 -> 1, drawn as a funnel with labelled columns."""
    sizes = [(7, "2,130"), (5, "256"), (4, "128"), (1, "1")]
    xs = [cx + (i - 1.5) * w / 3 for i in range(4)]
    cols = []
    for (m, _), x in zip(sizes, xs):
        step = h / max(m, 2)
        cols.append([(x, cy + (j - (m - 1) / 2) * step) for j in range(m)])
    for i in range(3):
        for pa in cols[i]:
            for pb in cols[i + 1]:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=GREY, lw=0.45,
                        zorder=2)
    for col in cols:
        for p in col:
            ax.scatter(*p, s=20, color=BLUE, zorder=3, linewidths=0)
    for (m, lab), x in zip(sizes, xs):
        ax.text(x, cy - h / 2 - 1.9, lab, ha="center", fontsize=8.5,
                color=SUB)


def box(x0, y0, w, h):
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white",
                           edgecolor=SUB, lw=1.1, zorder=1))


def arrow(a_, b_, color, lw=1.4):
    ax.add_patch(FancyArrowPatch(a_, b_, arrowstyle="-|>", mutation_scale=11,
                                 lw=lw, color=color, shrinkA=1, shrinkB=1,
                                 zorder=1))


# ------------------------------------------------------------------ inputs
table(11, 38, 13, 10)
ax.text(11, 30.5, f"{S['rows']} measurements", ha="center", fontsize=11,
        color=INK)
ax.text(11, 28.2, "2,130 columns", ha="center", fontsize=9, color=SUB)

molecule(11, 13, 5.5, ORANGE)
ax.text(11, 4.6, "relaxed complex", ha="center", fontsize=11, color=INK)

# ------------------------------------------------------------------ models
BX, BW, BH = 34, 27, 13
for y0 in (35.5, 18.5, 1.5):
    box(BX, y0, BW, BH)


def label(y0, title, r2):
    ax.text(BX + BW / 2, y0 + BH - 1.4, title, ha="center", va="top",
            fontsize=11, color=INK)
    ax.text(BX + BW - 1.2, y0 + 1.1, r2, ha="right", va="bottom",
            fontsize=9.5, color=SUB)


label(35.5, "boosted trees", f"R² {S['trees']}")
for dx in (-8.2, 0.0, 8.2):
    tree(BX + BW / 2 + dx, 40.6, 2.1)

label(18.5, "neural network", f"R² {S['fcnn']}")
net(BX + 10, 24.9, 15, 6.4)

label(1.5, "3D network", f"R² {S['enc']}")
molecule(BX + BW / 2, 6.6, 4.9, ORANGE, ms=10, mms=24)

# ------------------------------------------------------------------ arrows
arrow((17.8, 40), (BX, 42), BLUE)
arrow((17.8, 37), (BX, 25), BLUE)
arrow((17.8, 34), (BX, 10.5), BLUE)
arrow((17.5, 13), (BX, 6), ORANGE)

# ------------------------------------------------------------------ output
box(78, 18.5, 19, 13)
ax.text(87.5, 27.3, "combined", ha="center", fontsize=11, color=INK)
ax.text(87.5, 22.3, f"R² {S['comb']}", ha="center", fontsize=15, color=INK,
        fontweight="bold")
for y, yt in ((42, 28.0), (25, 25.0), (8, 22.0)):
    arrow((BX + BW, y), (78, yt), GREY)

fig.text(0.5, 0.03,
         "Adjacent-pair log SF R², leave-extractants-out "
         f"cross-validation, {S['pairs']} pairs.",
         ha="center", fontsize=9, color=SUB)

fig.savefig(HERE / S["out"], bbox_inches="tight")
print("wrote", S["out"])
