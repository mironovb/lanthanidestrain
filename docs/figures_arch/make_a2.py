#!/usr/bin/env python3
"""a2_evidence.png -- how the 3D shape weight is chosen, why the distance
encoder and not the simplicial one, and how the gain holds up.

Six panels, all from out-of-fold predictions:
 A  adjacent-pair R2 as a function of the 3D mixing weight, for both encoders
 B  the weight chosen per held-out extractant (nested selection)
 C  what each shape source contributes: correlation with the truth and with
    the tabular model's residual, plus the encoder-encoder correlation
 D  the gain across evaluation sets, including the held-out 444 pairs and
    the collaborator's expanded set
 E  independent seed halves of both ensembles
 F  representations that do not work in this slot
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "fig_data.json").read_text())

INK, SUB, GRID = "#101214", "#52514e", "#e3e5e8"
NAVY, BLUE, ORANGE, GREEN = "#1a2e4a", "#2a78d6", "#eb6834", "#1baf7a"
RED, GREY, CARD = "#c0392b", "#9aa1a8", "#f4f6f8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white",
    "savefig.dpi": 300, "axes.edgecolor": SUB, "axes.labelcolor": INK,
    "xtick.color": SUB, "ytick.color": SUB,
    "axes.spines.top": False, "axes.spines.right": False})


def guard(txt: str, width_in: float, fs: float) -> str:
    budget = int(width_in * 72 / (0.54 * fs))
    for ln in txt.split("\n"):
        assert len(ln) <= budget, f"too long ({len(ln)}>{budget}): {ln!r}"
    return txt


def tag(ax, letter, title):
    ax.set_title(f"{letter}   {title}", fontsize=10, color=NAVY,
                 loc="left", pad=7)


fig = plt.figure(figsize=(13.6, 8.2))
H1, H2 = 0.545, 0.080
HT = 0.305

# ------------------------------------------------------------------ A
axA = fig.add_axes([0.055, H1, 0.255, HT])
wc = D["weight_curve"]
w = np.array(wc["w"])
axA.plot(w, wc["dist"], "-", color=ORANGE, lw=2.4, label="distance encoder")
axA.plot(w, wc["snn"], color=GREY, lw=2.0, ls="--",
         label="simplicial encoder")
best = int(np.argmax(wc["dist"]))
axA.plot(w[best], wc["dist"][best], "o", color=ORANGE, ms=9,
         markeredgecolor="white", markeredgewidth=1.4, zorder=5)
axA.axhline(wc["dist"][0], color=BLUE, lw=1.2, ls=":")
axA.text(0.62, wc["dist"][0] - 0.004, "tabular shape only",
         fontsize=8.2, color=BLUE, va="top")
w_used = float(np.mean(D["nested_w_dist"]))
axA.axvline(w_used, color=INK, lw=1.2, ls="--", zorder=4)
axA.annotate(f"grid optimum {w[best]:.2f}\n{wc['dist'][best]:+.4f}",
             xy=(w[best], wc["dist"][best]), xytext=(0.44, 0.338),
             fontsize=8.2, color=ORANGE, fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0))
axA.text(w_used + 0.03, 0.256, guard(
    f"weight used: {w_used:.2f},\nfitted per fold", 1.6, 8.0),
    fontsize=8.0, color=INK, va="bottom")
axA.set_xlabel("weight on the 3D shape", fontsize=9.5)
axA.set_ylabel("adjacent-pair log SF R²", fontsize=9.5)
axA.set_ylim(0.24, 0.345)
axA.grid(color=GRID, zorder=0)
axA.legend(fontsize=8, frameon=False, loc="lower left")
tag(axA, "A", "the mixing weight has an interior optimum")

# ------------------------------------------------------------------ B
axB = fig.add_axes([0.395, H1, 0.235, HT])
nw = np.array(D["nested_w_dist"])
axB.hist(nw, bins=np.arange(0.20, 0.56, 0.05), color=ORANGE, alpha=0.85,
         edgecolor="white", zorder=3)
axB.axvline(nw.mean(), color=INK, lw=1.6, ls="--", zorder=4)
axB.text(nw.mean() - 0.012, axB.get_ylim()[1] * 0.55,
         f"mean {nw.mean():.2f}", fontsize=8.6, color=INK,
         fontweight="bold", ha="right")
axB.set_xlabel("weight chosen for a held-out extractant", fontsize=9.5)
axB.set_ylabel("extractants", fontsize=9.5)
axB.grid(color=GRID, axis="y", zorder=0)
axB.text(0.02, 0.98, guard(
    f"chosen on the other\n{len(nw)-1} extractants only;\n"
    f"range {nw.min():.2f}–{nw.max():.2f}", 1.9, 8.2),
    transform=axB.transAxes, fontsize=8.2, color=SUB, va="top")
tag(axB, "B", "the weight is stable across folds")

# ------------------------------------------------------------------ C
axC = fig.add_axes([0.715, H1, 0.255, HT])
cr = D["corr"]
labels = ["tabular\nshape", "3D shape\n(distance)", "3D shape\n(simplicial)"]
vals_dy = [cr["with_dy"][n] for n in ("tab", "dist", "snn")]
vals_res = [np.nan, cr["with_tab_resid"]["dist"], cr["with_tab_resid"]["snn"]]
xs = np.arange(3)
axC.bar(xs - 0.19, vals_dy, width=0.36, color=[BLUE, ORANGE, GREY], zorder=3)
axC.bar(xs + 0.19, vals_res, width=0.36, color=[BLUE, ORANGE, GREY],
        alpha=0.45, zorder=3, hatch="///")
for x, v in zip(xs, vals_dy):
    axC.text(x - 0.19, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.2,
             color=INK)
for x, v in zip(xs, vals_res):
    if np.isfinite(v):
        axC.text(x + 0.19, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.2,
                 color=INK)
axC.set_xticks(xs); axC.set_xticklabels(labels, fontsize=8.4)
axC.set_ylabel("correlation", fontsize=9.5)
axC.set_ylim(0, 1.02)
axC.grid(color=GRID, axis="y", zorder=0)
axC.text(0.02, 0.98, guard(
    "solid:   correlation with the measured separation\n"
    "hatched: correlation with what the tabular model misses",
    3.35, 7.6), transform=axC.transAxes, fontsize=7.6, color=SUB, va="top")
axC.text(0.02, 0.79, guard(
    "the two encoders predict almost the same\n"
    f"thing:  r = {cr['matrix'][1][2]:.3f}", 3.35, 8.4),
    transform=axC.transAxes, fontsize=8.4, color=RED, va="top",
    fontweight="bold")
tag(axC, "C", "why the distance encoder takes the weight")

# ------------------------------------------------------------------ D
axD = fig.add_axes([0.055, H2, 0.365, HT])
cf = D["confirm"]["results"]
groups = [("legacy\n905 pairs", cf["legacy"]["tabular"]["r2"],
           cf["legacy"]["blend"]["r2"]),
          ("held out\n444 pairs", cf["fresh"]["tabular"]["r2"],
           cf["fresh"]["blend"]["r2"]),
          ("all\n1,349 pairs", cf["all"]["tabular"]["r2"],
           cf["all"]["blend"]["r2"]),
          ("collaborator\n1,220 pairs", D["collab"]["all"][0],
           D["collab"]["all"][1]),
          ("collaborator\nnew 345", D["collab"]["new"][0],
           D["collab"]["new"][1])]
xs = np.arange(len(groups))
tabv = [g[1] for g in groups]; blv = [g[2] for g in groups]
axD.bar(xs - 0.2, tabv, width=0.38, color=BLUE, zorder=3, label="tabular only")
axD.bar(xs + 0.2, blv, width=0.38, color=GREEN, zorder=3, label="+ 3D shape")
for x, t, b in zip(xs, tabv, blv):
    axD.text(x + 0.2, b + 0.006, f"{b-t:+.3f}", ha="center", fontsize=8.2,
             color=GREEN, fontweight="bold")
axD.set_xticks(xs)
axD.set_xticklabels([g[0] for g in groups], fontsize=8.2)
axD.set_ylabel("adjacent-pair log SF R²", fontsize=9.5)
axD.set_ylim(0, 0.40)
axD.grid(color=GRID, axis="y", zorder=0)
axD.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
tag(axD, "D", "the 3D gain is positive on every evaluation set")

# ------------------------------------------------------------------ E
axE = fig.add_axes([0.495, H2, 0.175, HT])
ss = D["seed_splits"]
xs = np.arange(len(ss))
tv = [s["tabular_only"]["r2"] for s in ss]
bv = [s["blend"]["r2"] for s in ss]
axE.bar(xs - 0.2, tv, width=0.38, color=BLUE, zorder=3)
axE.bar(xs + 0.2, bv, width=0.38, color=GREEN, zorder=3)
for x, t, b in zip(xs, tv, bv):
    axE.text(x + 0.2, b + 0.004, f"{b-t:+.3f}", ha="center", fontsize=8.2,
             color=GREEN, fontweight="bold")
axE.set_xticks(xs)
axE.set_xticklabels([f"seed half {i+1}" for i in range(len(ss))], fontsize=8.2)
axE.set_ylabel("adjacent-pair R²", fontsize=9.5)
axE.set_ylim(0.26, 0.35)
axE.grid(color=GRID, axis="y", zorder=0)
tag(axE, "E", "independent seeds")

# ------------------------------------------------------------------ F
axF = fig.add_axes([0.745, H2, 0.225, HT])
p = D["persistence"]
ref = D["systems"]["anchored"]["r2"]
rows = [("simplicial encoder\nin the shape slot",
         D["topo"]["part1_encoder_comparison"]["blend_snn4"]["r2"], GREY),
        ("+ 22 persistence\nstatistics", p["anch_g9_ens4"], RED),
        ("+ 279 persistence\nimage pixels", p["anch_g11_ens4"], RED),
        ("same statistics,\nblock means only", p["anch_g9_bm_ens4"], ORANGE)]
ys = np.arange(len(rows))[::-1]
axF.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6,
         zorder=3)
axF.axvline(ref, color=GREEN, lw=1.8, ls="--", zorder=4)
axF.set_ylim(-0.62, len(rows) - 0.05)
axF.text(ref, len(rows) - 0.32, f"reference {ref:+.3f}", fontsize=8.2,
         color=GREEN, fontweight="bold", va="center", ha="center")
for yy, r in zip(ys, rows):
    if r[1] > 0.15:                      # label inside the bar
        axF.text(r[1] - 0.012, yy, f"{r[1]:+.3f}", va="center", ha="right",
                 fontsize=8.2, color="white", fontweight="bold", zorder=5)
    else:
        axF.text(r[1] + 0.012, yy, f"{r[1]:+.3f}", va="center", ha="left",
                 fontsize=8.2, color=INK, zorder=5)
axF.set_yticks(ys)
axF.set_yticklabels([r[0] for r in rows], fontsize=8.0)
axF.set_xlabel("adjacent-pair log SF R²", fontsize=9.5)
axF.set_xlim(-0.10, 0.42)
axF.grid(color=GRID, axis="x", zorder=0)
tag(axF, "F", "what fails in the same slot")

fig.text(0.055, 0.965, "How the 3D shape weight is chosen, and how far the "
         "gain travels", fontsize=15, color=NAVY, fontweight="bold")
fig.text(0.055, 0.930, guard(
    "Weights are fitted on the training extractants only; every score is "
    "out-of-fold under leave-extractants-out cross-validation.\n"
    "The held-out 444 pairs took no part in any earlier choice.",
    13.2, 10.2), fontsize=10.2, color=SUB, va="top")
fig.text(0.055, 0.018, guard(
    "F: persistence features are given to the shape model only. Replacing "
    "them by their within-block means recovers 78 % of the loss, so the "
    "damage is their across-metal variation inside a block.", 13.2, 8.4),
    fontsize=8.4, color=SUB)

fig.savefig(HERE / "a2_evidence.png")
print("wrote a2_evidence.png")
