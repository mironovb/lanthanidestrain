#!/usr/bin/env python3
"""a1_architecture.png -- what the current best 3D system computes.

Panel A: the data flow, with the real column counts, model settings and the
fitted mixing weight.  Panel B/C: the same arithmetic executed on one real
14-metal block (out-of-fold predictions), separating the level from the
shape.  Panel D: why the split matters -- 87 % of log D variance is level,
and the scored metric reads none of it.

All numbers come from docs/figures_arch/fig_data.json (built from the
out-of-fold parquets); nothing here is illustrative.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "fig_data.json").read_text())

INK, SUB, GRID = "#101214", "#52514e", "#e3e5e8"
NAVY, BLUE, ORANGE, GREEN = "#1a2e4a", "#2a78d6", "#eb6834", "#1baf7a"
GREY, CARD, EDGE = "#9aa1a8", "#f4f6f8", "#d5dadf"

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


def box(ax, x0, y0, w, h, title, lines, color, fs_t=10.5, fs_b=8.6):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.4",
                                linewidth=1.5, edgecolor=color,
                                facecolor=CARD, zorder=2))
    ax.text(x0 + w / 2, y0 + h - 3.2, title, ha="center", va="top",
            fontsize=fs_t, color=color, fontweight="bold", zorder=3)
    ax.text(x0 + w / 2, y0 + h - 8.6, "\n".join(lines), ha="center", va="top",
            fontsize=fs_b, color=SUB, zorder=3, linespacing=1.5)


def arrow(ax, xy_from, xy_to, color=GREY, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                 mutation_scale=13, linewidth=lw,
                                 color=color, zorder=1,
                                 shrinkA=2, shrinkB=2))


fig = plt.figure(figsize=(13.6, 9.4))

# ---------------------------------------------------------------- panel A
axA = fig.add_axes([0.035, 0.435, 0.93, 0.485])
axA.set_xlim(0, 100); axA.set_ylim(0, 100); axA.axis("off")

box(axA, 3, 80, 44, 18, "tabular inputs", [
    f"{D['n_rows']:,} measurements · 746 columns",
    "extractant fingerprint + conditions + metal"], BLUE)
box(axA, 53, 80, 44, 18, "3D inputs", [
    "956 Architector / GFN2-xTB complexes",
    "atoms, bonds, interatomic distances ≤ 4 Å"], ORANGE)

box(axA, 3, 50, 28, 21, "level model", [
    "CatBoost, quantile 0.6", "target: log D",
    "depth 9 · rsm 0.3"], BLUE, fs_t=10)
box(axA, 36, 50, 28, 21, "shape model", [
    "CatBoost, same settings", "target: log D − block mean",
    "sees only within-block signal"], BLUE, fs_t=10)
box(axA, 69, 50, 28, 21, "distance encoder", [
    "message passing on distances", "pair-contrast loss (weight 4)",
    f"{D['n_seeds_dist']}-seed ensemble"], ORANGE, fs_t=10)

for x0 in (14, 47):
    arrow(axA, (25, 80), (x0 + 3, 71.5), color=BLUE)
arrow(axA, (75, 80), (83, 71.5), color=ORANGE)

box(axA, 3, 22, 28, 18, "anchor", [
    "block mean of the", "level model's predictions",
    "(constant inside a block)"], BLUE, fs_t=10, fs_b=8.4)
box(axA, 36, 22, 28, 18, "tabular shape", [
    "shape model minus", "its own block mean"], BLUE, fs_t=10, fs_b=8.4)
box(axA, 69, 22, 28, 18, "3D shape", [
    "encoder minus", "its own block mean"], ORANGE, fs_t=10, fs_b=8.4)

for x in (17, 50, 83):
    arrow(axA, (x, 50), (x, 40.5), color=GREY)

box(axA, 20, 1, 60, 15, "prediction", [
    "anchor  +  0.65 × tabular shape  +  0.35 × 3D shape",
    "mixing weight fitted per held-out extractant"], GREEN,
    fs_t=11, fs_b=9.4)
arrow(axA, (17, 22), (33, 16.5), color=GREY)
arrow(axA, (50, 22), (50, 16.5), color=GREY)
arrow(axA, (83, 22), (67, 16.5), color=GREY)

axA.text(1.5, 99, "A", fontsize=14, fontweight="bold", color=INK)
axA.text(99, 8, guard(
    "the anchor cancels in every\n"
    "scored comparison: it is the\n"
    "same for both metals of a pair", 2.9, 8.4),
    ha="right", va="center", fontsize=8.4, color=SUB, style="italic")

# ------------------------------------------------------------- panels B,C
ex = D["example"]
m, y = ex["metals"], np.array(ex["y"])
xs = np.arange(len(m))

axB = fig.add_axes([0.055, 0.075, 0.29, 0.285])
axB.axhline(ex["anchor"], color=BLUE, ls="--", lw=1.4, zorder=2)
axB.vlines(xs, ex["anchor"], y, color=GREY, lw=1.0, zorder=1)
axB.plot(xs, y, "o", color=INK, ms=5.5, zorder=3, label="measured log D")
axB.set_xticks(xs); axB.set_xticklabels(m, fontsize=7.5)
axB.set_ylabel("log D", fontsize=9.5)
axB.grid(color=GRID, axis="y", zorder=0)
axB.text(0.4, ex["anchor"] + 0.12, f"anchor = {ex['anchor']:+.2f}",
         fontsize=8.4, color=BLUE, fontweight="bold")
axB.annotate("", xy=(11, y[11]), xytext=(11, ex["anchor"]),
             arrowprops=dict(arrowstyle="<->", color=SUB, lw=1.1))
axB.text(11.35, (y[11] + ex["anchor"]) / 2, "shape", fontsize=8.4,
         color=SUB, va="center")
axB.set_title("B   one real block: level vs shape", fontsize=10,
              color=NAVY, loc="left", pad=6)

axC = fig.add_axes([0.395, 0.075, 0.29, 0.285])
sy = np.array(ex["shape_y"]); stb = np.array(ex["shape_tab"])
sen = np.array(ex["shape_enc"]); bl = 0.65 * stb + 0.35 * sen
axC.axhline(0, color=SUB, lw=0.9)
axC.plot(xs, sy, "o-", color=INK, ms=4.5, lw=1.4, label="measured", zorder=4)
axC.plot(xs, stb, "s--", color=BLUE, ms=3.6, lw=1.2, label="tabular shape",
         zorder=3)
axC.plot(xs, sen, "^--", color=ORANGE, ms=3.6, lw=1.2, label="3D shape",
         zorder=3)
axC.plot(xs, bl, "-", color=GREEN, lw=2.2, label="blend (0.65/0.35)", zorder=2)
axC.set_xticks(xs); axC.set_xticklabels(m, fontsize=7.5)
axC.set_ylabel("log D − block mean", fontsize=9.5)
axC.grid(color=GRID, axis="y", zorder=0)
axC.legend(fontsize=7.6, frameon=False, loc="upper left", ncol=2,
           columnspacing=1.0, handlelength=1.6)
axC.set_title("C   the shape channel, same block", fontsize=10,
              color=NAVY, loc="left", pad=6)

# ---------------------------------------------------------------- panel D
axD = fig.add_axes([0.745, 0.075, 0.225, 0.285])
v = D["variance"]
lvl = v["level_share"] * 100
axD.bar([0], [lvl], color=BLUE, width=0.62, zorder=3)
axD.bar([0], [100 - lvl], bottom=[lvl], color=ORANGE, width=0.62, zorder=3)
axD.set_xlim(-0.6, 1.55); axD.set_ylim(0, 116)
axD.set_xticks([]); axD.set_ylabel("share of log D variance (%)", fontsize=9.5)
axD.text(0, lvl / 2, f"level\n{lvl:.0f} %", ha="center", va="center",
         fontsize=10.5, color="white", fontweight="bold")
axD.text(0, lvl + (100 - lvl) / 2, f"shape  {100-lvl:.0f} %", ha="center",
         va="center", fontsize=9, color="white", fontweight="bold")
axD.annotate("", xy=(0.42, 100), xytext=(0.42, lvl),
             arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4))
axD.text(0.52, (lvl + 100) / 2, guard(
    "the scored metric\nreads only this\npart", 1.5, 8.6),
    fontsize=8.6, color=ORANGE, va="center", fontweight="bold")
axD.text(0.52, lvl / 2, guard(
    f"between-block\nspread sd {v['level_sd']:.2f}\n"
    f"cancels on\ndifferencing", 1.6, 8.2),
    fontsize=8.2, color=SUB, va="center")
axD.grid(color=GRID, axis="y", zorder=0)
axD.set_title("D   why the split matters", fontsize=10, color=NAVY,
              loc="left", pad=6)

fig.text(0.035, 0.972, "The current best system: predict the level and the "
         "shape separately, and put 3D only in the shape",
         fontsize=15, color=NAVY, fontweight="bold")
fig.text(0.035, 0.938, guard(
    f"Out-of-fold, leave-extractants-out ({D['n_extractants']} extractants, "
    f"{D['n_blocks']} blocks). Adjacent-pair log SF R²: flat model "
    f"{D['systems']['flat']['r2']:+.3f} → level/shape split "
    f"{D['systems']['anchored']['r2']:+.3f} → with the 3D shape "
    f"{D['systems']['blend']['r2']:+.3f}.", 13.4, 10.2),
    fontsize=10.2, color=SUB)
fig.text(0.035, 0.022, guard(
    "Panels B/C: block \"" + ex["ligand"][:34] + "…\", "
    f"{len(m)} lanthanides, out-of-fold predictions. "
    "The 3D encoder never sees the level: its predictions are centred "
    "inside each block before use.", 13.4, 8.4),
    fontsize=8.4, color=SUB)

fig.savefig(HERE / "a1_architecture.png")
print("wrote a1_architecture.png")
