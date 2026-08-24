#!/usr/bin/env python3
"""Encoder pipeline: from the finished Vietoris-Rips mesh to a predicted log D.

Saves figure.png (300 dpi) and figure.svg in this directory.
Every quantity is taken from automl/topo/snn.py, automl/topo/simplicial_data.py
and automl/topo/train.py; see the notes accompanying this script.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

HERE = Path(__file__).resolve().parent

INK = "#1A1A1A"
MID = "#6E6E6E"
LIGHT = "#B4B4B4"
FILL = "#E6E6E6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
})

FIG_W, FIG_H = 14.3, 5.2                      # 11 : 4
fig = plt.figure(figsize=(FIG_W, FIG_H))

ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 120)
ax.set_ylim(26, 100)
ax.axis("off")

ROW_Y0, ROW_Y1 = 44.0, 68.0                   # main-row box extent
MID_Y = 0.5 * (ROW_Y0 + ROW_Y1)


def box(x0, x1, title, lines, y0=ROW_Y0, y1=ROW_Y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white",
                           edgecolor=INK, linewidth=1.0, zorder=2))
    xc = 0.5 * (x0 + x1)
    ax.text(xc, y1 - 5.0, title, ha="center", va="center", fontsize=9,
            color=INK, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(xc, y1 - 11.5 - 6.0 * i, ln, ha="center", va="center",
                fontsize=7.5, color=MID, zorder=3)


def arrow(x0, x1, y=MID_Y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=9))


# ---------------------------------------------------------------- 1. mesh
axm = fig.add_axes([0.012, 0.205, 0.105, 0.40])
axm.set_aspect("equal")
axm.axis("off")

ang_d = np.deg2rad(np.arange(0, 360, 60))
ang_o = np.deg2rad(np.arange(30, 390, 60))
pts = np.vstack([[0.0, 0.0],
                 np.column_stack([2.4 * np.cos(ang_d), 2.4 * np.sin(ang_d)]),
                 np.column_stack([4.2 * np.cos(ang_o), 4.2 * np.sin(ang_o)])])
CUT = 3.5
dm = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
tri = [t for t in combinations(range(len(pts)), 3)
       if dm[t[0], t[1]] <= CUT and dm[t[0], t[2]] <= CUT
       and dm[t[1], t[2]] <= CUT]
axm.add_patch(Polygon(pts[list(tri[0])], closed=True, facecolor=FILL,
                      edgecolor="none", zorder=1))
for i, j in combinations(range(len(pts)), 2):
    if dm[i, j] <= CUT:
        axm.plot(pts[[i, j], 0], pts[[i, j], 1], color=LIGHT, linewidth=0.7,
                 zorder=2)
axm.scatter(pts[1:, 0], pts[1:, 1], s=16, color=MID, zorder=3)
axm.scatter([0], [0], s=34, color=INK, zorder=4)
lim = 4.9
axm.set_xlim(-lim, lim)
axm.set_ylim(-lim, lim)

ax.text(6.5, 37.0, "Vietoris–Rips complex", ha="center", va="center",
        fontsize=8, color=INK)
ax.text(6.5, 31.5, "at 3.5 Å (2-skeleton)", ha="center", va="center",
        fontsize=7.5, color=MID)

# ------------------------------------------------------------- 2. inputs
arrow(13.0, 15.5)
box(15.5, 31.0, "inputs",
    ["per atom: 5 values + element", "per edge, triangle: its length"])

# ----------------------------------------------------- 3. message passing
arrow(31.0, 34.0)
box(34.0, 49.5, "message passing",
    ["3 rounds", "along edges and triangles"])

# ------------------------------------------------------------ 4. read-out
arrow(49.5, 52.5)
box(52.5, 69.0, "metal-centred read-out",
    ["soft histogram of d(metal)", "32 bins, 0–8 Å; atoms, donors"])

# ----------------------------------------------------------- 5. embedding
arrow(69.0, 72.0)
BAR_X0, BAR_X1 = 72.0, 88.0
BAR_Y0, BAR_Y1 = 52.0, 60.0
blocks = ["node mean", "node max", "edge mean", "edge max", "triangle mean",
          "triangle max", "metal atom", "metal edges", "radial shells"]
seg = (BAR_X1 - BAR_X0) / len(blocks)
for i, name in enumerate(blocks):
    x0 = BAR_X0 + i * seg
    ax.add_patch(Rectangle((x0, BAR_Y0), seg, BAR_Y1 - BAR_Y0,
                           facecolor="white", edgecolor=INK, linewidth=0.7,
                           zorder=2))
    ax.text(x0 + 0.5 * seg, BAR_Y0 - 1.6, name, ha="center", va="top",
            fontsize=6.2, color=MID, rotation=90, zorder=3)
ax.text(0.5 * (BAR_X0 + BAR_X1), 66.5, "embedding: 9 blocks × 96 = 864",
        ha="center", va="center", fontsize=8, color=INK)
ax.text(0.5 * (BAR_X0 + BAR_X1), 61.8, "computed once per complex",
        ha="center", va="bottom", fontsize=7, color=MID)

# ---------------------------------------------------------------- 6. join
arrow(88.0, 91.0)
box(91.0, 99.5, "concat", ["1,610"])

box(78.0, 106.0, "746 tabular columns of the row",
    ["descriptors, fingerprints, 64 conditions"], y0=76.0, y1=93.0)
ax.annotate("", xy=(95.25, 68.0), xytext=(95.25, 76.0),
            arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.0,
                            shrinkA=0, shrinkB=0, mutation_scale=9))

# ---------------------------------------------------------------- 7. head
arrow(99.5, 102.5)
box(102.5, 114.0, "head",
    ["LayerNorm, SiLU", "1610 → 256 → 128 → 1"])

arrow(114.0, 116.5)
ax.text(117.2, MID_Y, "log D", ha="left", va="center", fontsize=9, color=INK)

fig.savefig(HERE / "figure.png", dpi=300, bbox_inches="tight")
fig.savefig(HERE / "figure.svg", bbox_inches="tight")
print("wrote figure.png and figure.svg")
