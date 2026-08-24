#!/usr/bin/env python3
"""Vietoris-Rips complex of one real lanthanide complex, and how it enters
the model.

A: heavy atoms of one GFN2-xTB structure (Dy complex, build 2b190032c56e)
   projected to 2D, with the 1- and 2-simplices present at r = 3.5 A.
B: the same complex at three filtration radii.
C: the model path from the complex to the prediction.

Usage:  PYTHONPATH=$PWD python3 docs/talk_acs/make_vr_fig.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = "2b190032c56e"

INK, GREY, LINE = "#111111", "#7d838a", "#c9cdd2"
ORANGE, BLUE, RED, FILL = "#d55e00", "#2a6db0", "#c0392b", "#f4f6f8"
ELEM = {6: ("#8a8f94", "C"), 7: ("#2a6db0", "N"), 8: ("#c0392b", "O"),
        15: ("#b8860b", "P"), 16: ("#c9a227", "S")}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "figure.facecolor": "white", "savefig.dpi": 400})

# ------------------------------------------------------------------ data
z = np.load(REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz")
bids = [str(b) for b in z["build_ids"]]
k = bids.index(BUILD)
a, b = int(z["node_ptr"][k]), int(z["node_ptr"][k + 1])
Z = z["atomic_numbers"][a:b]
xyz = z["coordinates"][a:b].astype(float)
ism = z["is_metal"][a:b].astype(bool)
isd = z["is_coord_donor"][a:b].astype(bool)
heavy = Z > 1
Z, xyz, ism, isd = Z[heavy], xyz[heavy], ism[heavy], isd[heavy]
n = len(Z)
c = xyz - xyz.mean(0)
u, s, vt = np.linalg.svd(c, full_matrices=False)
p2 = c @ vt[:2].T
dm = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)


def simplices(r):
    e = [(i, j) for i, j in combinations(range(n), 2) if dm[i, j] <= r]
    t = [(i, j, l) for i, j, l in combinations(range(n), 3)
         if dm[i, j] <= r and dm[i, l] <= r and dm[j, l] <= r]
    return e, t


def draw(ax, r, node_size, edge_lw, label_atoms=False, tri=True):
    e, t = simplices(r)
    if tri:
        for i, j, l in t:
            ax.add_patch(Polygon(p2[[i, j, l]], closed=True,
                                 facecolor=ORANGE, alpha=0.10,
                                 edgecolor="none", zorder=1))
    for i, j in e:
        ax.plot(p2[[i, j], 0], p2[[i, j], 1], color=LINE, lw=edge_lw,
                zorder=2)
    for i in range(n):
        if ism[i]:
            ax.scatter(*p2[i], s=node_size * 3.2, color=ORANGE,
                       edgecolor=INK, linewidth=0.7, zorder=4)
        else:
            col = ELEM.get(int(Z[i]), (GREY, ""))[0]
            ax.scatter(*p2[i], s=node_size, color=col,
                       edgecolor=INK if isd[i] else "none",
                       linewidth=0.8 if isd[i] else 0, zorder=3)
    ax.set_aspect("equal"); ax.axis("off")
    lim = np.abs(p2).max() * 1.12
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    return len(e), len(t)


fig = plt.figure(figsize=(7.0, 4.2))

# ------------------------------------------------------------------ A
axA = fig.add_axes([0.02, 0.31, 0.40, 0.64])
ne, nt = draw(axA, 3.5, 34, 0.7)
axA.text(0.0, 1.0, "A", transform=axA.transAxes, fontsize=11,
         fontweight="bold", va="top")
axA.text(0.5, 0.005, f"r = 3.5 Å:  {n} heavy atoms · {ne} edges · {nt} triangles",
         transform=axA.transAxes, ha="center", va="bottom", fontsize=7.6,
         color=GREY)
# legend
lx, ly = 0.80, 0.70
for dy_, (col, lab, fmt) in enumerate([
        (ORANGE, "Ln³⁺", "m"), (RED, "O", "o"), (BLUE, "N", "o"),
        (GREY, "C", "o")]):
    yy = ly + dy_ * 0.055
    axA.scatter(lx, yy, transform=axA.transAxes, s=30 if fmt == "m" else 16,
                color=col, edgecolor=INK if fmt == "m" else "none",
                linewidth=0.6, zorder=6, clip_on=False)
    axA.text(lx + 0.05, yy, lab, transform=axA.transAxes, fontsize=7.5,
             va="center")
axA.scatter(lx, ly + 4 * 0.055, transform=axA.transAxes, s=16,
            facecolor="white", edgecolor=INK, linewidth=0.8, clip_on=False)
axA.text(lx + 0.05, ly + 4 * 0.055, "donor atom", transform=axA.transAxes,
         fontsize=7.5, va="center")

# ------------------------------------------------------------------ B
axB_tag = fig.add_axes([0.45, 0.30, 0.53, 0.66])
axB_tag.axis("off")
axB_tag.text(0.0, 1.0, "B", transform=axB_tag.transAxes, fontsize=11,
             fontweight="bold", va="top")
for i, r in enumerate((2.0, 2.8, 3.5)):
    ax = fig.add_axes([0.46 + i * 0.175, 0.42, 0.165, 0.42])
    ne, nt = draw(ax, r, 16, 0.55)
    ax.text(0.5, -0.03, f"r = {r:.1f} Å\n{ne} edges\n{nt} triangles",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.4,
            color=GREY, linespacing=1.3)
axB_tag.text(0.08, 0.985, "edge (i, j) present when d(i, j) ≤ r;\n"
             "triangle when all three of its edges are present",
             transform=axB_tag.transAxes, ha="left", va="top",
             fontsize=7.6, color=INK, linespacing=1.3)

# ------------------------------------------------------------------ C
axC = fig.add_axes([0.02, 0.02, 0.96, 0.25])
axC.set_xlim(0, 100); axC.set_ylim(0, 100); axC.axis("off")
axC.text(0, 100, "C", fontsize=11, fontweight="bold", va="top")


def cbox(x, w, title, sub, color=INK):
    axC.add_patch(FancyBboxPatch((x, 14), w, 70,
                                 boxstyle="round,pad=0.5,rounding_size=1.2",
                                 facecolor=FILL, edgecolor=color,
                                 linewidth=0.9))
    axC.text(x + w / 2, 69, title, ha="center", va="center", fontsize=7.8,
             color=color, fontweight="bold")
    axC.text(x + w / 2, 40, sub, ha="center", va="center", fontsize=6.6,
             color=GREY, linespacing=1.25)


def carrow(x1, x2):
    axC.add_patch(FancyArrowPatch((x1, 51), (x2, 51), arrowstyle="-|>",
                                  mutation_scale=8, linewidth=0.9,
                                  color=INK))


cbox(2, 24, "node features",
     "element, partial charge,\nmetal / donor flag,\ndistance to the metal")
carrow(26.3, 28.7)
cbox(29, 24, "message passing",
     "along edges (distance encoder)\nor along edges and triangles\n(simplicial network)")
carrow(53.3, 55.7)
cbox(56, 18, "metal-shell readout", "atoms within the\ncoordination sphere")
carrow(74.3, 76.7)
cbox(77, 21, "log D per measurement",
     "centred within its block,\nthen shape channel\n(weight 0.35)", ORANGE)

fig.savefig(HERE / "vr_complex.png")
print(f"wrote vr_complex.png  ({n} heavy atoms)")
