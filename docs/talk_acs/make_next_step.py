#!/usr/bin/env python3
"""The step after the mesh: how the network reads it.

Two panels in the style of the mesh-construction slide.
Saves next_step.png (300 dpi) and next_step.svg.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = "2b190032c56e"
CUT = 3.5

INK = "#1A1A1A"
NODE = "#808080"
EDGE = "#C8C8C8"
TRI = "#EDEDED"
CAP = "#333333"
SUB = "#777777"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

# --------------------------------------------------------------- geometry
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
dm = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
edges = [(i, j) for i, j in combinations(range(n), 2) if dm[i, j] <= CUT]
tris = [t for t in combinations(range(n), 3)
        if dm[t[0], t[1]] <= CUT and dm[t[0], t[2]] <= CUT
        and dm[t[1], t[2]] <= CUT]
metal = int(np.argmax(ism))
nbr = {i: [j for j in range(n) if j != i and dm[i, j] <= CUT] for i in range(n)}

# a non-metal atom whose neighbours fan out, so the arrows read clearly
def spread(i):
    ang = np.sort(np.arctan2(*(P[nbr[i]] - P[i]).T[::-1]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
    return gaps.min()


cand = [i for i in range(n) if i != metal and 4 <= len(nbr[i]) <= 7]
focus = max(cand, key=spread)


def mesh(ax, faint=False):
    for t in tris:
        ax.add_patch(Polygon(P[list(t)], closed=True, facecolor=TRI,
                             edgecolor="none", zorder=1))
    for i, j in edges:
        ax.plot(P[[i, j], 0], P[[i, j], 1], color=EDGE, lw=1.2, zorder=2,
                solid_capstyle="round")
    ax.scatter(P[:, 0], P[:, 1], s=52, color=NODE, zorder=3, linewidths=0)
    ax.scatter(*P[metal], s=86, color=INK, zorder=4, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    lim = np.abs(P).max() * 1.18
    ax.set_xlim(-lim, lim * 1.02)
    ax.set_ylim(-lim, lim)


fig = plt.figure(figsize=(9.4, 4.2))

# --------------------------------------------------- A: what an atom holds
axA = fig.add_axes([0.02, 0.08, 0.44, 0.76])
mesh(axA)
fig.text(0.24, 0.93, "every atom carries a few numbers", ha="center",
         va="center", fontsize=13.5, color=CAP)
fig.text(0.24, 0.045, "charge · metal/donor flags · distance to Ln · element",
         ha="center", va="center", fontsize=10, color=SUB)

# ------------------------------------------------------ B: message passing
axB = fig.add_axes([0.52, 0.08, 0.44, 0.76])
mesh(axB)
for t in tris:
    if focus in t:
        axB.add_patch(Polygon(P[list(t)], closed=True, facecolor="#DCDCDC",
                              edgecolor="none", zorder=1.5))
for j in nbr[focus]:
    v = P[focus] - P[j]
    L = np.linalg.norm(v)
    axB.annotate("", xy=P[focus] - v / L * 0.42, xytext=P[j] + v / L * 0.42,
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0,
                                 mutation_scale=8, shrinkA=0, shrinkB=0),
                 zorder=6)
axB.scatter(P[nbr[focus], 0], P[nbr[focus], 1], s=52, color="#4D4D4D",
            zorder=5, linewidths=0)
axB.scatter(*P[focus], s=86, color=INK, zorder=6, linewidths=0)
fig.text(0.74, 0.93, "each atom updates from its neighbours", ha="center",
         va="center", fontsize=13.5, color=CAP)
fig.text(0.74, 0.045, "three rounds — three steps out along the mesh",
         ha="center", va="center", fontsize=10, color=SUB)

fig.savefig(HERE / "next_step.png", dpi=300, bbox_inches="tight")
fig.savefig(HERE / "next_step.svg", bbox_inches="tight")
print(f"wrote next_step.png / .svg  (focus atom degree {len(nbr[focus])})")
