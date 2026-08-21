#!/usr/bin/env python3
"""Talk figures for the ACS presentation.

Eight projection-sized figures (12.4 x 5.6 in, 300 dpi, large fonts), all
built from the same out-of-fold artefacts as the report figures.  Titles are
deliberately omitted: each slide's title carries the assertion, the figure
carries only the evidence.

  T1_scoring       the level/shape split of the target
  T2_architecture  the system, as a diagram
  T3_block         the shape channel on one real block
  T4_scoreboard    system progression on the scored metric
  T5_weight        how the 3D weight is chosen; encoder redundancy
  T6_confirm       held-out and cross-population checks
  T7_negatives     representations that fail, with the block-mean control
  T8_collab        independent reproduction and the head-to-head

Usage:  PYTHONPATH=$PWD python3 docs/talk_acs/make_talk_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D = json.loads((REPO / "docs/figures_arch/fig_data.json").read_text())

INK, SUB, GRID = "#101214", "#52514e", "#e3e5e8"
NAVY, BLUE, ORANGE, GREEN = "#1a2e4a", "#2a78d6", "#eb6834", "#1baf7a"
RED, GREY, CARD = "#c0392b", "#9aa1a8", "#f4f6f8"
W3D = 0.35
FIGSIZE = (12.4, 5.6)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white",
    "savefig.dpi": 300, "axes.edgecolor": SUB, "axes.labelcolor": INK,
    "xtick.color": SUB, "ytick.color": SUB, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False})


def guard(txt, width_in, fs):
    budget = int(width_in * 72 / (0.54 * fs))
    for ln in txt.split("\n"):
        assert len(ln) <= budget, f"too long ({len(ln)}>{budget}): {ln!r}"
    return txt


def panel_tag(ax, letter, title, fs=13):
    ax.set_title(f"{letter}   {title}", fontsize=fs, color=NAVY,
                 loc="left", pad=9)


def save(fig, name):
    fig.savefig(HERE / f"{name}.png")
    plt.close(fig)
    print(f"  {name}.png")


# =====================================================================  T1
def t1_scoring():
    fig = plt.figure(figsize=FIGSIZE)
    ex = D["example"]
    m, y = ex["metals"], np.array(ex["y"])
    xs = np.arange(len(m))

    ax = fig.add_axes([0.075, 0.145, 0.415, 0.75])
    ax.axhline(ex["anchor"], color=BLUE, ls="--", lw=2.0, zorder=2)
    ax.vlines(xs, ex["anchor"], y, color=GREY, lw=1.4, zorder=1)
    ax.plot(xs, y, "o", color=INK, ms=8, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(m, fontsize=11)
    ax.set_ylabel("measured  log D", fontsize=13)
    ax.grid(color=GRID, axis="y", zorder=0)
    ax.text(0.35, ex["anchor"] + 0.14, "block level", fontsize=12.5,
            color=BLUE, fontweight="bold")
    ax.annotate("", xy=(11, y[11]), xytext=(11, ex["anchor"]),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.0))
    ax.text(11.4, (y[11] + ex["anchor"]) / 2, "shape", fontsize=12.5,
            color=ORANGE, va="center", fontweight="bold")
    panel_tag(ax, "A", "one extractant, one condition set: 14 lanthanides")

    axv = fig.add_axes([0.565, 0.145, 0.085, 0.75])
    v = D["variance"]; lvl = v["level_share"] * 100
    axv.bar([0], [lvl], color=BLUE, width=0.75, zorder=3)
    axv.bar([0], [100 - lvl], bottom=[lvl], color=ORANGE, width=0.75, zorder=3)
    axv.set_xlim(-0.55, 0.55); axv.set_ylim(0, 100)
    axv.set_xticks([])
    axv.set_ylabel("share of log D variance (%)", fontsize=12.5)
    axv.text(0, lvl / 2, f"level\n{lvl:.0f} %", ha="center", va="center",
             fontsize=13, color="white", fontweight="bold")
    axv.text(0, lvl + (100 - lvl) / 2, f"shape\n{100 - lvl:.0f} %",
             ha="center", va="center", fontsize=11, color="white",
             fontweight="bold")
    axv.grid(color=GRID, axis="y", zorder=0)
    panel_tag(axv, "B", "", fs=13)

    fig.text(0.70, 0.56, guard(
        "The separation factor is a\n"
        "difference inside a block,\n"
        "so the level cancels exactly.\n\n"
        "A model that spends its\n"
        "capacity on the 87 % is\n"
        "optimising the wrong thing.", 3.5, 12.5),
        fontsize=12.5, color=INK, va="center")
    save(fig, "T1_scoring")


# =====================================================================  T2
def t2_architecture():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x0, y0, w, h, title, lines, color, fs_t=13, fs_b=11):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                    boxstyle="round,pad=0.7,rounding_size=1.6",
                                    linewidth=2.0, edgecolor=color,
                                    facecolor=CARD, zorder=2))
        ax.text(x0 + w / 2, y0 + h - 4.5, title, ha="center", va="top",
                fontsize=fs_t, color=color, fontweight="bold", zorder=3)
        ax.text(x0 + w / 2, y0 + h - 12, "\n".join(lines), ha="center",
                va="top", fontsize=fs_b, color=SUB, zorder=3, linespacing=1.6)

    def arrow(a, b, color=GREY):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                                     mutation_scale=18, linewidth=2.0,
                                     color=color, zorder=1,
                                     shrinkA=3, shrinkB=3))

    box(2, 76, 46, 22, "tabular inputs",
        ["4,746 measurements × 746 columns",
         "fingerprint · conditions · metal"], BLUE)
    box(54, 76, 44, 22, "3D inputs",
        ["956 GFN2-xTB complexes",
         "distances ≤ 4 Å"], ORANGE)

    box(2, 40, 29, 26, "level model",
        ["CatBoost", "target: log D"], BLUE)
    box(35.5, 40, 29, 26, "shape model",
        ["CatBoost", "target: log D −", "block mean"], BLUE)
    box(69, 40, 29, 26, "distance encoder",
        ["graph network,", "contrast loss,", "32-seed ensemble"], ORANGE)

    arrow((18, 76), (16, 66.5), BLUE)
    arrow((32, 76), (50, 66.5), BLUE)
    arrow((76, 76), (83, 66.5), ORANGE)

    ax.text(16.5, 34, "block mean", ha="center", fontsize=11.5, color=BLUE,
            style="italic")
    ax.text(50, 34, "centre in block", ha="center", fontsize=11.5, color=BLUE,
            style="italic")
    ax.text(83.5, 34, "centre in block", ha="center", fontsize=11.5,
            color=ORANGE, style="italic")
    for x, c in ((16.5, BLUE), (50, BLUE), (83.5, ORANGE)):
        arrow((x, 40), (x, 36), c)
        arrow((x, 31), (x, 22.5), c)

    box(8, 4, 84, 18, "prediction",
        ["anchor   +   0.65 × tabular shape   +   0.35 × 3D shape"],
        GREEN, fs_t=14, fs_b=13.5)
    save(fig, "T2_architecture")


# =====================================================================  T3
def t3_block():
    fig = plt.figure(figsize=FIGSIZE)
    ex = D["example"]
    m = ex["metals"]; xs = np.arange(len(m))
    sy = np.array(ex["shape_y"]); stb = np.array(ex["shape_tab"])
    sen = np.array(ex["shape_enc"]); bl = (1 - W3D) * stb + W3D * sen

    ax = fig.add_axes([0.065, 0.145, 0.545, 0.76])
    ax.axhline(0, color=SUB, lw=1.2)
    ax.plot(xs, sy, "o-", color=INK, ms=7, lw=2.0, label="measured", zorder=4)
    ax.plot(xs, stb, "s--", color=BLUE, ms=5.5, lw=1.6,
            label="tabular shape", zorder=3)
    ax.plot(xs, sen, "^--", color=ORANGE, ms=5.5, lw=1.6,
            label="3D shape", zorder=3)
    ax.plot(xs, bl, "-", color=GREEN, lw=3.0, label="blend", zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels(m, fontsize=11)
    ax.set_ylabel("log D − block mean", fontsize=13)
    ax.grid(color=GRID, axis="y", zorder=0)
    ax.legend(fontsize=11.5, frameon=False, ncol=2, loc="upper left",
              columnspacing=1.4)
    panel_tag(ax, "A", "the shape channel")

    axB = fig.add_axes([0.70, 0.145, 0.28, 0.76])
    # only TRUE adjacent pairs: Pm is absent from the data, so Nd-Sm is a
    # two-step gap and the metric excludes it
    idx = np.array(ex["idx"])
    keep = [i for i in range(len(m) - 1) if idx[i + 1] - idx[i] == 1]
    d_true = np.array([sy[i + 1] - sy[i] for i in keep])
    d_bl = np.array([bl[i + 1] - bl[i] for i in keep])
    lab = [f"{m[i]}–{m[i+1]}" for i in keep]
    ys = np.arange(len(lab))[::-1]
    axB.barh(ys + 0.19, d_true, height=0.36, color=INK, zorder=3,
             label="measured")
    axB.barh(ys - 0.19, d_bl, height=0.36, color=GREEN, zorder=3,
             label="predicted")
    axB.axvline(0, color=SUB, lw=1.2)
    axB.set_yticks(ys); axB.set_yticklabels(lab, fontsize=10)
    axB.set_xlabel("adjacent separation  (log SF)", fontsize=12.5)
    axB.grid(color=GRID, axis="x", zorder=0)
    axB.legend(fontsize=11, frameon=False, loc="lower right")
    panel_tag(axB, "B", "what is scored")
    save(fig, "T3_block")


# =====================================================================  T4
def t4_scoreboard():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.30, 0.16, 0.66, 0.76])
    rows = [
        ("3D encoder alone", D["systems"]["encoder_alone"]["r2"], ORANGE),
        ("gradient boosting, flat", D["systems"]["flat"]["r2"], GREY),
        ("July 2026 three-model stack", 0.2672, GREY),
        ("best stack before this work", 0.3132, BLUE),
        ("level/shape split", D["systems"]["anchored"]["r2"], NAVY),
        ("level/shape split + 3D shape", D["systems"]["blend"]["r2"], GREEN),
    ]
    ys = np.arange(len(rows))
    ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows],
            height=0.62, zorder=3)
    for yy, r in zip(ys, rows):
        ax.text(r[1] + 0.004, yy, f"{r[1]:+.3f}", va="center", fontsize=12.5,
                color=INK, fontweight="bold" if r[2] in (GREEN, NAVY) else
                "normal")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=12.5)
    ax.set_xlabel("adjacent-pair log SF  R²   (out-of-fold, extractants held out)",
                  fontsize=12.5)
    ax.set_xlim(0, 0.425)
    ax.grid(color=GRID, axis="x", zorder=0)
    ax.axvline(0.3132, color=BLUE, ls=":", lw=1.6, zorder=2)
    ax.annotate(guard("beats the best\ncombination", 1.5, 12),
                xy=(D["systems"]["blend"]["r2"] + 0.004, ys[-1] - 0.42),
                xytext=(0.345, ys[-1] - 2.35), fontsize=12, color=GREEN,
                fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8,
                                connectionstyle="arc3,rad=-0.15"))
    save(fig, "T4_scoreboard")


# =====================================================================  T5
def t5_weight():
    fig = plt.figure(figsize=FIGSIZE)
    wc = D["weight_curve"]; w = np.array(wc["w"])
    nw = np.array(D["nested_w_dist"]); cr = D["corr"]

    ax = fig.add_axes([0.070, 0.155, 0.255, 0.72])
    ax.plot(w, wc["dist"], color=ORANGE, lw=3.0, label="distance encoder")
    ax.plot(w, wc["snn"], color=GREY, lw=2.4, ls="--",
            label="simplicial encoder")
    ax.axhline(wc["dist"][0], color=BLUE, lw=1.6, ls=":")
    ax.text(0.50, wc["dist"][0] - 0.005, "no 3D", fontsize=11.5, color=BLUE,
            va="top")
    b = int(np.argmax(wc["dist"]))
    ax.plot(w[b], wc["dist"][b], "o", color=ORANGE, ms=11,
            markeredgecolor="white", markeredgewidth=1.8, zorder=5)
    ax.set_xlabel("weight on the 3D shape", fontsize=12.5)
    ax.set_ylabel("adjacent-pair R²", fontsize=12.5)
    ax.set_ylim(0.24, 0.345)
    ax.grid(color=GRID, zorder=0)
    ax.legend(fontsize=11, frameon=False, loc="lower left")
    panel_tag(ax, "A", "an interior optimum")

    axB = fig.add_axes([0.395, 0.155, 0.22, 0.72])
    axB.hist(nw, bins=np.arange(0.20, 0.56, 0.05), color=ORANGE,
             edgecolor="white", zorder=3)
    axB.axvline(nw.mean(), color=INK, lw=2.0, ls="--", zorder=4)
    axB.text(nw.mean() - 0.015, axB.get_ylim()[1] * 0.55,
             f"mean {nw.mean():.2f}", fontsize=12, color=INK,
             fontweight="bold", ha="right")
    axB.set_xlabel("weight chosen per held-out\nextractant", fontsize=12.5)
    axB.set_ylabel("extractants", fontsize=12.5)
    axB.grid(color=GRID, axis="y", zorder=0)
    panel_tag(axB, "B", "stable across folds")

    axC = fig.add_axes([0.70, 0.155, 0.275, 0.72])
    labs = ["tabular", "3D\ndistance", "3D\nsimplicial"]
    vd = [cr["with_dy"][n] for n in ("tab", "dist", "snn")]
    vr = [np.nan, cr["with_tab_resid"]["dist"], cr["with_tab_resid"]["snn"]]
    xs = np.arange(3)
    axC.bar(xs - 0.19, vd, width=0.36, color=[BLUE, ORANGE, GREY], zorder=3)
    axC.bar(xs + 0.19, vr, width=0.36, color=[BLUE, ORANGE, GREY],
            alpha=0.45, hatch="///", zorder=3)
    for x, v in zip(xs, vd):
        axC.text(x - 0.19, v + 0.015, f"{v:.2f}", ha="center", fontsize=11)
    for x, v in zip(xs, vr):
        if np.isfinite(v):
            axC.text(x + 0.19, v + 0.015, f"{v:.2f}", ha="center", fontsize=11)
    axC.set_xticks(xs); axC.set_xticklabels(labs, fontsize=11.5)
    axC.set_ylabel("correlation", fontsize=12.5)
    axC.set_ylim(0, 0.95)
    axC.grid(color=GRID, axis="y", zorder=0)
    axC.text(0.5, 0.97, guard(
        "solid: with the measured separation\n"
        "hatched: with what tabular misses", 3.2, 10.5),
        transform=axC.transAxes, fontsize=10.5, color=SUB, ha="center",
        va="top")
    axC.text(0.5, 0.79, guard(
        f"the two encoders agree at r = {cr['matrix'][1][2]:.3f}", 3.2, 11.5),
        transform=axC.transAxes, fontsize=11.5, color=RED, ha="center",
        va="top", fontweight="bold")
    panel_tag(axC, "C", "why distance, not simplicial")
    save(fig, "T5_weight")


# =====================================================================  T6
def t6_confirm():
    fig = plt.figure(figsize=FIGSIZE)
    cf = D["confirm"]["results"]
    ax = fig.add_axes([0.075, 0.20, 0.455, 0.70])
    groups = [("legacy\n905 pairs", cf["legacy"]),
              ("HELD OUT\n444 pairs", cf["fresh"]),
              ("all\n1,349", cf["all"])]
    extra = [("collaborator\n1,220", D["collab"]["all"]),
             ("collaborator\nnew 345", D["collab"]["new"])]
    labels = [g[0] for g in groups] + [e[0] for e in extra]
    tabv = [g[1]["tabular"]["r2"] for g in groups] + [e[1][0] for e in extra]
    blv = [g[1]["blend"]["r2"] for g in groups] + [e[1][1] for e in extra]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, tabv, width=0.38, color=BLUE, zorder=3,
           label="tabular only")
    ax.bar(xs + 0.2, blv, width=0.38, color=GREEN, zorder=3,
           label="+ 3D shape")
    for x, t, b in zip(xs, tabv, blv):
        ax.text(x + 0.2, b + 0.008, f"{b - t:+.3f}", ha="center",
                fontsize=11.5, color=GREEN, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylabel("adjacent-pair log SF R²", fontsize=12.5)
    ax.set_ylim(0, 0.40)
    ax.grid(color=GRID, axis="y", zorder=0)
    ax.legend(fontsize=11.5, frameon=False, loc="upper right", ncol=2)
    panel_tag(ax, "A", "every evaluation set improves")

    axB = fig.add_axes([0.615, 0.20, 0.115, 0.70])
    ss = D["seed_splits"]
    xs2 = np.arange(len(ss))
    tv = [s["tabular_only"]["r2"] for s in ss]
    bv = [s["blend"]["r2"] for s in ss]
    axB.bar(xs2 - 0.2, tv, width=0.38, color=BLUE, zorder=3)
    axB.bar(xs2 + 0.2, bv, width=0.38, color=GREEN, zorder=3)
    for x, t, b in zip(xs2, tv, bv):
        axB.text(x + 0.2, b + 0.003, f"{b - t:+.3f}", ha="center",
                 fontsize=10.5, color=GREEN, fontweight="bold")
    axB.set_xticks(xs2)
    axB.set_xticklabels([f"half {i+1}" for i in range(len(ss))], fontsize=11)
    axB.set_ylabel("adjacent-pair R²", fontsize=12)
    axB.set_ylim(0.26, 0.35)
    axB.grid(color=GRID, axis="y", zorder=0)
    panel_tag(axB, "B", "seed halves")

    fig.text(0.775, 0.55, guard(
        "The 444 held-out pairs\n"
        "were frozen before any\n"
        "model was trained, and\n"
        "the mixing weight was\n"
        "fixed in advance.", 2.7, 12),
        fontsize=12, color=INK, va="center")
    save(fig, "T6_confirm")


# =====================================================================  T7
def t7_negatives():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.31, 0.16, 0.65, 0.76])
    ref = D["systems"]["anchored"]["r2"]
    topo = D["topo"]["part1_encoder_comparison"]
    p = D["persistence"]
    rows = [
        ("simplicial encoder in the shape slot",
         topo["blend_snn4"]["r2"], GREY),
        ("+ 22 persistence statistics", p["anch_g9_ens4"], RED),
        ("+ 279 persistence-image pixels", p["anch_g11_ens4"], RED),
        ("+ both persistence blocks", p["anch_g9_g11_ens4"], RED),
        ("same statistics, block means only", p["anch_g9_bm_ens4"], ORANGE),
    ]
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows],
            height=0.6, zorder=3)
    ax.axvline(ref, color=GREEN, lw=2.4, ls="--", zorder=4)
    ax.axvline(0, color=SUB, lw=1.2, zorder=2)
    for yy, r in zip(ys, rows):
        if r[1] > 0.15:
            ax.text(r[1] - 0.008, yy, f"{r[1]:+.3f}", va="center", ha="right",
                    fontsize=12, color="white", fontweight="bold", zorder=5)
        else:
            ax.text(r[1] + 0.008, yy, f"{r[1]:+.3f}", va="center", ha="left",
                    fontsize=12, color=INK, zorder=5)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlabel("adjacent-pair log SF R²", fontsize=12.5)
    ax.set_xlim(-0.12, 0.42)
    ax.set_ylim(-0.75, len(rows) - 0.15)
    ax.grid(color=GRID, axis="x", zorder=0)
    ax.text(ref - 0.012, len(rows) - 0.42, f"reference  {ref:+.3f}",
            fontsize=12, color=GREEN, fontweight="bold", ha="right")
    ax.annotate(guard("removing the within-block\n"
                      "variation recovers 78 %", 3.0, 11.5),
                xy=(p["anch_g9_bm_ens4"], ys[-1]),
                xytext=(0.02, ys[-1] - 0.68), fontsize=11.5, color=ORANGE,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
    save(fig, "T7_negatives")


# =====================================================================  T8
def t8_collab():
    fig = plt.figure(figsize=FIGSIZE)
    L = pd.read_csv(REPO / "automl/reports/collab_ours/leaderboard_ours.csv")
    rep = pd.read_csv(REPO / "automl/reports/collab_repro/leaderboard.csv")

    ax = fig.add_axes([0.075, 0.19, 0.28, 0.70])
    arms = ["A2", "A2_TP", "PAIRMEAN"]
    his = [0.3192, 0.3175, 0.4482]
    ours = [float(rep[rep.arm == a]["macro_mae"].iloc[0]) for a in arms]
    xs = np.arange(len(arms))
    ax.bar(xs - 0.2, his, width=0.38, color=GREY, zorder=3,
           label="his reported")
    ax.bar(xs + 0.2, ours, width=0.38, color=BLUE, zorder=3,
           label="our reproduction")
    for x, h, o in zip(xs, his, ours):
        ax.text(x + 0.2, o + 0.009, f"{o:.3f}", ha="center", fontsize=9.5)
        ax.text(x - 0.2, h + 0.009, f"{h:.3f}", ha="center", fontsize=9.5,
                color=SUB)
    ax.set_xticks(xs)
    ax.set_xticklabels(["his\nmodel", "his model\n+ post-proc.", "baseline"],
                       fontsize=11)
    ax.set_ylabel("macro MAE  (log SF, lower is better)", fontsize=12.5)
    ax.set_ylim(0, 0.55)
    ax.grid(color=GRID, axis="y", zorder=0)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    panel_tag(ax, "A", "reproduced from his spec alone")

    axB = fig.add_axes([0.435, 0.19, 0.265, 0.70])
    order = ["A2", "A2_TP", "OURS_anchored", "OURS_anchored3D"]
    names = ["his\nmodel", "his\n+ post", "ours\nsplit", "ours\n+ 3D"]
    vals = [float(L[L.arm == a]["macro_mae"].iloc[0]) for a in order]
    cols = [GREY, GREY, NAVY, GREEN]
    xs2 = np.arange(len(order))
    axB.bar(xs2, vals, color=cols, width=0.62, zorder=3)
    for x, v in zip(xs2, vals):
        axB.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=11.5)
    axB.set_xticks(xs2); axB.set_xticklabels(names, fontsize=11.5)
    axB.set_ylabel("macro MAE", fontsize=12.5)
    axB.set_ylim(0, 0.42)
    axB.grid(color=GRID, axis="y", zorder=0)
    panel_tag(axB, "B", "his cohort, his metric")

    fig.text(0.725, 0.55, guard(
        "Two independent\n"
        "pipelines, one metric:\n"
        "every difference has a\n"
        "confidence interval\n"
        "that spans zero.\n\n"
        "Ours runs under the\n"
        "stricter split.", 3.2, 12),
        fontsize=12, color=INK, va="center")
    save(fig, "T8_collab")


if __name__ == "__main__":
    print("building talk figures:")
    t1_scoring(); t2_architecture(); t3_block(); t4_scoreboard()
    t5_weight(); t6_confirm(); t7_negatives(); t8_collab()
    print("done")
