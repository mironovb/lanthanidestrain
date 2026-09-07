#!/usr/bin/env python3
"""contraction.png -- the lanthanide-contraction benchmark of two Hamiltonians.

Data: automl/artifacts/gxtb_series/cf_shard{0,1}.json (gas phase, 71 ligands
x 15 lanthanides x 2 Hamiltonians) and compliance_test.json.  c_L is the
slope of the mean metal-donor distance on the Shannon (1976) radius, fitted
over Ce..Lu (La is a separate GFN2 parameter anchor and is excluded), exactly
as automl/qc/compliance_test.py computes it.  c_L = 1 means the computed
donor shell contracts exactly as the experimental ionic radius does.

A  one ligand, the one whose GFN2 slope is the median of the 71
B  all 71 ligands, GFN2 and g-xTB paired
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SER = REPO / "automl/artifacts/gxtb_series"

INK, SUB, GREY, LIGHT = "#1a1a1a", "#666666", "#b0b0b0", "#d9d9d9"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
ARM = {"gfn2": ("GFN2-xTB", BLUE), "gxtb_hs": ("g-xTB", ORANGE),
       "lnxtb_hs": ("Ln-xTB", GREEN)}          # Ln-xTB drawn when its arm exists

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white",
    "savefig.dpi": 300, "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out"})

SHANNON = {
    8: {"La": 1.160, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109, "Pm": 1.093,
        "Sm": 1.079, "Eu": 1.066, "Gd": 1.053, "Tb": 1.040, "Dy": 1.027,
        "Ho": 1.015, "Er": 1.004, "Tm": 0.994, "Yb": 0.985, "Lu": 0.977},
    9: {"La": 1.216, "Ce": 1.196, "Pr": 1.179, "Nd": 1.163, "Pm": 1.144,
        "Sm": 1.132, "Eu": 1.120, "Gd": 1.107, "Tb": 1.095, "Dy": 1.083,
        "Ho": 1.072, "Er": 1.062, "Tm": 1.052, "Yb": 1.042, "Lu": 1.032}}

# ---------------------------------------------------------------- data
recs = []
for t in ("cf_shard0", "cf_shard1", "lnxtb_shard0", "lnxtb_shard1"):
    if not (SER / f"{t}.json").exists():
        continue
    d = json.loads((SER / f"{t}.json").read_text())
    recs += [r for r in d["records"] if r.get("ok")]
df = pd.DataFrame(recs)
df = df[df.f_count >= 1]                       # drops La, as compliance_test does
comp = pd.DataFrame(json.loads((SER / "compliance_test.json").read_text())["compliance"])
cw = comp.pivot(index="family", columns="arm", values="c_L")
ARM = {a: v for a, v in ARM.items() if a in cw.columns}
cw = cw[list(ARM)].dropna()
assert len(cw) == 71, len(cw)
g2, gx = cw["gfn2"].to_numpy(), cw["gxtb_hs"].to_numpy()
series = [(cw[a].to_numpy(), col) for a, (_, col) in ARM.items()]
print(f"71 ligands · GFN2 {g2.mean():.3f} ± {g2.std(ddof=1):.3f} · "
      f"g-xTB {gx.mean():.3f} ± {gx.std(ddof=1):.3f} · "
      f"g-xTB > GFN2 on {(gx > g2).sum()} of {len(cw)}")

# example ligand: GFN2 slope at the median of the 71
fam = cw.index[np.argsort(g2)[len(g2) // 2]]
cn = int(comp[comp.family == fam].iloc[0]["cn"])
sh = SHANNON[cn]
print("example ligand:", fam.split("||")[0], "· CN", cn,
      f"· c_L GFN2 {cw.loc[fam,'gfn2']:.3f}, g-xTB {cw.loc[fam,'gxtb_hs']:.3f}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(10.4, 4.3))
axA = fig.add_axes([0.085, 0.16, 0.40, 0.74])
axB = fig.add_axes([0.60, 0.16, 0.37, 0.74])

# A: one ligand, both Hamiltonians ----------------------------------------
for arm, (name, col) in ARM.items():
    g = df[(df.family == fam) & (df.arm == arm)]
    x = g.metal.map(sh).to_numpy(float)
    y = g.mean_m_donor.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    c, b = np.polyfit(x, y, 1)
    xx = np.array([x.min() - 0.006, x.max() + 0.006])
    axA.plot(xx, c * xx + b, color=col, lw=1.4, zorder=2)
    axA.scatter(x, y, s=26, color=col, zorder=3, linewidths=0)
    axA.text(xx[0] - 0.005, c * xx[0] + b, f"{name}\nslope {c:.2f}",
             color=col, fontsize=9, va="center", ha="left", linespacing=1.25)
axA.set_xlabel("Shannon ionic radius of Ln³⁺  (Å)", fontsize=10)
axA.set_ylabel("mean Ln–donor distance  (Å)", fontsize=10)
axA.invert_xaxis()
x0, x1 = axA.get_xlim()
axA.set_xlim(x0, x1 - 0.06)
axA.text(-0.16, 1.03, "A", transform=axA.transAxes, fontsize=12,
         fontweight="bold", va="bottom")

# B: 71 ligands, paired ----------------------------------------------------
xs = np.arange(len(series), dtype=float)
rng = np.random.default_rng(0)
jit = rng.uniform(-0.07, 0.07, len(cw))
for xc, (v, col) in zip(xs, series):
    axB.scatter(xc + jit, v, s=16, color=col, zorder=3, linewidths=0)
    axB.plot([xc - 0.16, xc + 0.16], [v.mean()] * 2, color=col, lw=2.2,
             zorder=4)
    axB.text(xc, -0.02, f"{v.mean():.3f} ± {v.std(ddof=1):.3f}", ha="center",
             va="top", fontsize=8.5, color=col, transform=axB.get_xaxis_transform())
axB.axhline(1.0, color=INK, lw=0.8, ls="--", zorder=2)
axB.text(-0.42, 1.02, "Shannon radii", fontsize=8.5, color=INK,
         va="bottom", ha="left")
axB.set_xticks(xs)
axB.set_xticklabels([name for name, _ in ARM.values()], fontsize=10)
axB.tick_params(axis="x", length=0, pad=18)
axB.set_xlim(-0.45, xs[-1] + 0.75)
axB.set_ylim(0, 1.45)
axB.set_ylabel("contraction slope  c$_L$", fontsize=10)
axB.text(-0.17, 1.03, "B", transform=axB.transAxes, fontsize=12,
         fontweight="bold", va="bottom")

fig.savefig(HERE / "contraction.png", bbox_inches="tight")
fig.savefig(HERE / "contraction.pdf", bbox_inches="tight")
print("wrote contraction.png / .pdf")
