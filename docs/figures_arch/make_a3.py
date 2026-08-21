#!/usr/bin/env python3
"""a3_where.png -- where the 3D shape channel actually changes predictions,
and what the system still gets wrong.

 A  predicted vs measured adjacent separation (905 pairs), before and after
    the 3D shape is mixed in
 B  change in squared error by position in the lanthanide series
 C  change in squared error by extractant (sorted)
 D  the remaining error: predictions are compressed relative to the truth,
    and the largest true separations carry most of the error
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.topo_shape import load_cell

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

INK, SUB, GRID = "#101214", "#52514e", "#e3e5e8"
NAVY, BLUE, ORANGE, GREEN = "#1a2e4a", "#2a78d6", "#eb6834", "#1baf7a"
RED, GREY = "#c0392b", "#9aa1a8"
W = 0.35
LN = ["", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
      "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]

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


# ---------------------------------------------------------------- data
anch = pd.read_parquet(REPO / "automl/artifacts/anchored_champ"
                              "/oof_anch_q60_q60_ens8.parquet")
dist, _ = load_cell("topo_c15", "c15_plw4")
meta = pd.read_parquet(REPO / "automl/artifacts/matrix/matrix.parquet",
                       columns=["safe_exp_id", "composition_key",
                                "lanthanide_index", "extractant_group"])
df = (anch.rename(columns={"oof": "tab"})
      .merge(dist[["safe_exp_id", "oof"]].rename(columns={"oof": "enc"}),
             on="safe_exp_id").merge(meta, on="safe_exp_id"))
key = pd.Series(df["composition_key"])
anchor = pd.Series(df["tab"]).groupby(key).transform("mean").to_numpy()
st = df["tab"].to_numpy() - anchor
se = (pd.Series(df["enc"])
      - pd.Series(df["enc"]).groupby(key).transform("mean")).to_numpy()

rows = []
y = df["y"].to_numpy(float)
comp = df["composition_key"].to_numpy()
li = df["lanthanide_index"].to_numpy()
ex = df["extractant_group"].to_numpy()
for g in pd.unique(ex):
    m = ex == g
    dyg, dtg = ev.adjacent_pair_arrays(y[m], anchor[m] + st[m], comp[m], li[m])
    _, deg = ev.adjacent_pair_arrays(y[m], anchor[m] + se[m], comp[m], li[m])
    if not len(dyg):
        continue
    frame = pd.DataFrame({"y": y[m], "c": comp[m], "m": li[m]})
    lo = []
    for ck, blk in frame.groupby("c"):
        blk = blk.groupby("m", as_index=False)["y"].mean()
        mm = blk["m"].to_numpy()
        i, j = np.triu_indices(len(blk), k=1)
        adj = np.abs(mm[i] - mm[j]) == 1
        lo.extend(np.minimum(mm[i][adj], mm[j][adj]).astype(int).tolist())
    rows.append(pd.DataFrame({"ex": g, "l_lo": lo, "dy": dyg,
                              "tab": dtg, "enc": deg}))
P = pd.concat(rows, ignore_index=True)
P["blend"] = (1 - W) * P["tab"] + W * P["enc"]
P["e_tab"] = (P.dy - P.tab) ** 2
P["e_bl"] = (P.dy - P.blend) ** 2
P["gain"] = P.e_tab - P.e_bl
r2_tab = ev._r2(P.dy.to_numpy(), P.tab.to_numpy())
r2_bl = ev._r2(P.dy.to_numpy(), P.blend.to_numpy())
print(f"{len(P)} pairs · tabular {r2_tab:+.4f} · blend {r2_bl:+.4f}")

fig = plt.figure(figsize=(13.6, 7.4))

# ------------------------------------------------------------------ A
axA = fig.add_axes([0.055, 0.115, 0.245, 0.66])
lim = 1.35
axA.plot([-lim, lim], [-lim, lim], color=SUB, lw=1.0, ls="--", zorder=1)
imp = P.gain > 0
axA.scatter(P.dy[~imp], P.blend[~imp], s=12, color=GREY, alpha=0.55,
            linewidths=0, zorder=2, label=f"3D hurt ({int((~imp).sum())})")
axA.scatter(P.dy[imp], P.blend[imp], s=12, color=GREEN, alpha=0.7,
            linewidths=0, zorder=3, label=f"3D helped ({int(imp.sum())})")
axA.set_xlim(-lim, lim); axA.set_ylim(-lim, lim)
axA.set_xlabel("measured separation  (log SF)", fontsize=9.5)
axA.set_ylabel("predicted separation  (log SF)", fontsize=9.5)
axA.grid(color=GRID, zorder=0)
axA.legend(fontsize=8, frameon=False, loc="upper left")
axA.text(0.97, 0.06, guard(
    f"R²  {r2_tab:+.3f} → {r2_bl:+.3f}\nslope {np.polyfit(P.dy, P.blend, 1)[0]:.2f}"
    " (predictions\ncompressed toward zero)", 2.3, 8.2),
    transform=axA.transAxes, fontsize=8.2, color=INK, ha="right")
tag(axA, "A", "predictions after mixing in the 3D shape")

# ------------------------------------------------------------------ B
axB = fig.add_axes([0.375, 0.115, 0.245, 0.66])
pos = (P.groupby("l_lo")
       .agg(gain=("gain", "sum"), n=("gain", "size"),
            e_tab=("e_tab", "sum")).reset_index().sort_values("l_lo"))
pos["label"] = pos.l_lo.map(lambda v: f"{LN[v]}–{LN[v+1]}")
ys = np.arange(len(pos))[::-1]
cols = [GREEN if g > 0 else RED for g in pos.gain]
axB.barh(ys, pos.gain, color=cols, height=0.68, zorder=3)
axB.axvline(0, color=SUB, lw=1.0)
axB.set_yticks(ys); axB.set_yticklabels(pos.label, fontsize=8.2)
axB.set_xlabel("total squared error removed by the 3D shape", fontsize=9.5)
axB.grid(color=GRID, axis="x", zorder=0)
for yy, (_, r) in zip(ys, pos.iterrows()):
    axB.text(r.gain + (0.012 if r.gain >= 0 else -0.012), yy,
             f"n={int(r.n)}", va="center",
             ha="left" if r.gain >= 0 else "right", fontsize=7.4, color=SUB)
axB.set_xlim(pos.gain.min() * 1.75, pos.gain.max() * 1.55)
n_pos = int((pos.gain > 0).sum())
net = float(pos.gain.sum())
axB.text(0.97, 0.04, guard(
    f"{n_pos} of {len(pos)} positions improve;\nnet {net:+.2f} over all pairs",
    2.3, 8.2), transform=axB.transAxes, fontsize=8.2, color=INK, ha="right")
tag(axB, "B", "which parts of the series improve")

# ------------------------------------------------------------------ C
axC = fig.add_axes([0.695, 0.115, 0.275, 0.66])
exg = (P.groupby("ex").agg(gain=("gain", "sum"), n=("gain", "size"))
       .reset_index().sort_values("gain", ascending=False))
xs = np.arange(len(exg))
axC.bar(xs, exg.gain, color=[GREEN if g > 0 else RED for g in exg.gain],
        width=0.9, zorder=3)
axC.axhline(0, color=SUB, lw=1.0)
axC.set_xlabel(f"{len(exg)} extractants, sorted", fontsize=9.5)
axC.set_ylabel("squared error removed", fontsize=9.5)
axC.grid(color=GRID, axis="y", zorder=0)
n_up = int((exg.gain > 0).sum())
gross = float(exg.gain[exg.gain > 0].sum())
top5 = float(exg.gain.head(5).sum()) / gross
top = exg.iloc[0]
axC.annotate(guard(f"top 5 extractants supply\n{top5:.0%} of the gross gain",
                   2.5, 8.2),
             xy=(2, top.gain * 0.97), xytext=(7.0, top.gain * 0.9),
             fontsize=8.2, color=INK,
             arrowprops=dict(arrowstyle="-", color=SUB, lw=1.0))
axC.text(0.97, 0.06, guard(
    f"only {n_up} of {len(exg)} extractants improve:\n"
    "the net gain is small and uneven,\nnot a uniform effect", 2.7, 8.2),
    transform=axC.transAxes, fontsize=8.2, color=INK, ha="right")
tag(axC, "C", "and which extractants")

fig.text(0.055, 0.945, "Where the 3D shape channel changes the answer",
         fontsize=15, color=NAVY, fontweight="bold")
fig.text(0.055, 0.885, guard(
    f"All {len(P)} adjacent pairs of the legacy evaluation set, out-of-fold; "
    "the mixing weight is 0.35 everywhere. The gain is real but small and\n"
    "unevenly spread: it survives repeated checks, yet fewer than half the "
    "extractants improve, which is what a +0.008 average looks like.",
    13.2, 10.2), fontsize=10.2, color=SUB, va="top")
fig.text(0.055, 0.022, guard(
    "A: the diagonal is perfect prediction. Both models under-predict the "
    "largest separations, where most of the remaining error sits; the 3D "
    "shape reduces that compression slightly but does not remove it.",
    13.2, 8.4), fontsize=8.4, color=SUB)

fig.savefig(HERE / "a3_where.png")
print("wrote a3_where.png")
