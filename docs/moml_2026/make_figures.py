#!/usr/bin/env python3
"""The paper's two data figures, one PDF per panel.

Figure 1  a: predicted vs measured adjacent log SF, best system, 905 held-out
             pairs;   b: per-position ranking quality on held-out extractants.
Figure 2  a: one ligand's Ln-donor distance vs Shannon radius under three
             Hamiltonians;   b: per-ligand contraction compliance, 71 ligands.

All numbers are read from the repository artefacts; nothing is typed in.
    PYTHONPATH=$PWD python3 docs/moml_2026/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import figstyle as fs                                   # noqa: E402
import automl.evaluation as ev                          # noqa: E402
from automl.topo.decision_quality import pair_table     # noqa: E402
from automl.qc.compliance_test import SHANNON           # noqa: E402

SER = REPO / "automl/artifacts/gxtb_series"
LN = ["", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
HAM = [("gfn2", "GFN2-xTB"), ("gxtb_hs", "g-xTB"), ("lnxtb_hs", "Ln-xTB")]
HCOL = {"gfn2": fs.COLOR["gfn2"], "gxtb_hs": fs.COLOR["gxtb"], "lnxtb_hs": fs.COLOR["lnxtb"]}


def letter(ax, ch):
    """16 pt bold panel letter, outside the axes at the top left."""
    if ch:
        ax.text(-0.22, 1.02, ch, transform=ax.transAxes, fontsize=fs.TITLE,
                fontweight="bold", ha="left", va="bottom")


# ------------------------------------------------------------ figure 1a
def fig1a(ch=None):
    P = pair_table()
    dy, dp = P.dy.to_numpy(), P.dp.to_numpy()
    r = np.corrcoef(dy, dp)[0, 1]
    assert len(P) == 905, len(P)
    fig, ax = fs.figure(3.4, 3.4)
    m = float(max(np.abs(dy).max(), np.abs(dp).max())) + 0.15   # limits from the data
    lim = (-m, m)
    ax.plot(lim, lim, ls="--", lw=0.7, color=fs.OI["black"], zorder=1)
    ax.text(-0.78 * m, -0.78 * m + 0.10, "1:1", rotation=45, rotation_mode="anchor",
            ha="center", va="bottom", color=fs.OI["black"])
    ax.scatter(dy, dp, s=8, color=fs.COLOR["model"], alpha=0.5, linewidths=0, zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xticks([-2, -1, 0, 1, 2]); ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("measured log SF"); ax.set_ylabel("predicted log SF")
    ax.text(0.04, 0.96, f"r = {r:.2f}\n{len(P)} pairs\n{P.ex.nunique()} extractants held out",
            transform=ax.transAxes, va="top", ha="left", linespacing=1.4)
    letter(ax, ch)
    fs.finish(fig, "fig1a_parity")
    return r


# ------------------------------------------------------------ figure 1b
def fig1b(ch=None):
    D = json.loads((REPO / "automl/reports/decision_quality.json").read_text())
    R = D["ranking"]["by_position"]
    pos = list(R.keys())[::-1]                      # La-Ce at the top
    rho = np.array([R[p]["spearman"] for p in pos])
    hit = np.array([R[p]["top_quartile_hit_rate"] for p in pos])
    chance = D["ranking"]["chance_hit_rate"]
    fig, ax = fs.figure(3.8, 3.7)      # wider so the full name fits inside the axes
    y = np.arange(len(pos)); h = 0.38
    ax.barh(y + h / 2, rho, height=h, color=fs.COLOR["model"])
    ax.barh(y - h / 2, hit, height=h, color=fs.COLOR["hit"])
    ax.vlines(chance, -0.7, len(pos) - 1 + 0.8, ls="--", lw=0.7, color=fs.OI["black"], zorder=3)
    # names written beside the shortest row's bars, where the axes are empty
    top = len(pos) - 1
    # names beside the first two rows: "Spearman ρ" centred on the La-Ce row
    # after its longer bar, the full hit-rate name centred on the Ce-Pr row;
    # chance at the top of its line
    top = len(pos) - 1
    k1, k2 = pos.index("La-Ce"), pos.index("Ce-Pr")
    ax.text(max(rho[k1], hit[k1]) + 0.03, k1, "Spearman ρ", va="center", ha="left", color=fs.COLOR["model"])
    ax.text(max(rho[k2], hit[k2]) + 0.03, k2, "top-quartile hit rate", va="center", ha="left", color=fs.COLOR["hit"])
    ax.text(chance, top + 0.85, "chance", ha="center", va="bottom", color=fs.OI["black"])
    ax.set_yticks(y); ax.set_yticklabels(pos)
    ax.set_ylim(-0.7, top + 1.7)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("ranking of held-out extractants")
    ax.tick_params(axis="y", length=0)
    letter(ax, ch)
    fs.finish(fig, "fig1b_ranking")


# ------------------------------------------------------------ figure 2 data
def series_records():
    recs = []
    for t in ("cf_shard0", "cf_shard1", "lnxtb_shard0", "lnxtb_shard1"):
        d = json.loads((SER / f"{t}.json").read_text())
        recs += [r for r in d["records"] if r.get("ok")]
    df = pd.DataFrame(recs)
    return df[df.f_count >= 1]                       # La excluded, as in the fits


def fig2a(df, comp):
    cw = comp.pivot(index="family", columns="arm", values="c_L")[[h for h, _ in HAM]].dropna()
    g2 = cw["gfn2"].to_numpy()
    fam = cw.index[np.argsort(g2)[len(g2) // 2]]     # the ligand at GFN2's median
    cn = int(comp[comp.family == fam].iloc[0]["cn"]); sh = SHANNON.get(cn, SHANNON[9])
    fig, ax = fs.figure(3.4, 3.4)
    for arm, name in HAM:
        g = df[(df.family == fam) & (df.arm == arm)]
        x = g.metal.map(sh).to_numpy(float); yv = g.mean_m_donor.to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(yv); x, yv = x[ok], yv[ok]
        c, b = np.polyfit(x, yv, 1)
        xx = np.array([x.min() - 0.005, x.max() + 0.005])
        ax.plot(xx, c * xx + b, color=HCOL[arm], lw=1.3, zorder=2)
        ax.scatter(x, yv, s=22, color=HCOL[arm], linewidths=0, zorder=3, label=name)
        ax.text(xx[0] - 0.004, c * xx[0] + b, f"{c:.2f}", color=HCOL[arm],
                ha="left", va="center")                 # slope at the line's small-radius end
    ax.invert_xaxis()
    x0, x1 = ax.get_xlim(); ax.set_xlim(x0, x1 - 0.03)   # room for the slope labels
    ax.set_xlabel("Shannon radius of Ln$^{3+}$ (Å)"); ax.set_ylabel("mean Ln–donor distance (Å)")
    fs.legend_above(ax, ncol=3)
    fs.finish(fig, "fig2a_series")
    return fam.split("||")[0], cn


def fig2b(comp, ch=None):
    cw = comp.pivot(index="family", columns="arm", values="c_L")[[h for h, _ in HAM]].dropna()
    assert len(cw) == 71, len(cw)
    fig, ax = fs.figure(3.4, 3.4)
    rng = np.random.default_rng(0); jit = rng.uniform(-0.09, 0.09, len(cw))
    for i, (arm, name) in enumerate(HAM):
        v = cw[arm].to_numpy()
        ax.scatter(i + jit, v, s=14, color=HCOL[arm], linewidths=0, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, color=HCOL[arm], lw=2.2, zorder=4)
    ax.axhline(1.0, ls="--", lw=0.9, color=fs.NEUTRAL, zorder=2, label="Shannon radii (1.00)")
    ax.set_xticks(range(len(HAM))); ax.set_xticklabels([n for _, n in HAM])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(HAM) - 0.5); ax.set_ylim(0, 1.5)
    ax.set_ylabel("contraction compliance c$_L$")
    fs.legend_above(ax, ncol=1)
    letter(ax, ch)
    fs.finish(fig, "fig2b_compliance")
    return {n: (float(cw[a].mean()), float(cw[a].std(ddof=1))) for a, n in HAM}


# ------------------------------------------------------------ figure 3
def fig3(ch=None):
    """Learning curve over training extractants (automl/topo/learning_curve.py)."""
    arms = [("both", "level and shape models", fs.COLOR["model"]),
            ("shape_only", "shape model only", fs.COLOR["hit"])]
    data = {}
    for arm, _, _ in arms:
        p = REPO / f"automl/reports/learning_curve_{arm}.json"
        if p.exists():
            data[arm] = json.loads(p.read_text())
    if "both" not in data:
        print("fig3: no learning-curve results yet"); return None
    fig, ax = fs.figure(3.4, 3.4)
    F = data["both"]["fits"]
    pts = pd.DataFrame(F["points"]); col = fs.COLOR["model"]
    n_full = float(pts.n.max())
    h1 = ax.errorbar(pts.n, pts.r2, yerr=[pts.r2 - pts.lo, pts.hi - pts.r2], fmt="o", color=col,
                     ms=5, capsize=2.5, lw=1.1, label="mean of 4 seeds × 3 draws (range)")
    xx = np.linspace(pts.n.min() * 0.9, 2.0 * n_full, 200)
    hy, pw = F["hyperbolic"], F["power"]
    h2, = ax.plot(xx, hy["R_inf"] - hy["k"] / xx, color=col, lw=1.1, ls="-", label="hyperbolic fit")
    h3, = ax.plot(xx, pw["R_inf"] - pw["k"] * xx ** (-pw["c"]), color=col, lw=1.1, ls=":",
                  label="power-law fit")
    h4 = ax.axvline(n_full, ls="--", lw=0.9, color=fs.NEUTRAL, label="all training extractants")
    ax.set_xscale("log"); ax.set_xlabel("training extractants with adjacent pairs")
    ax.set_ylabel("adjacent-pair log SF R²")
    ax.set_xticks([10, 20, 40, 80, 120]); ax.set_xticklabels(["10", "20", "40", "80", "120"])
    ax.minorticks_off()
    fs.legend_above(ax, [h1, h2, h3, h4], [h.get_label() for h in (h1, h2, h3, h4)], ncol=1)
    letter(ax, ch)
    fs.finish(fig, "fig3_learning_curve")
    return data["both"]["fits"]["verdict"]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    only = ap.parse_args().only
    if only is not None:
        if "a" in only: print("fig1a r =", round(fig1a("a"), 3))
        if "b" in only: fig1b("b")
        if "c" in only:
            comp = pd.DataFrame(json.loads((SER / "compliance_test.json").read_text())["compliance"]); fig2b(comp, "c")
        if "d" in only: fig3("d")
        return 0
    r = fig1a("a"); fig1b("b")
    df = series_records()
    comp = pd.DataFrame(json.loads((SER / "compliance_test.json").read_text())["compliance"])
    lig, cn = fig2a(df, comp); stats = fig2b(comp, "c")
    print(f"fig1a r = {r:.3f}; fig2a example ligand {lig} (CN {cn}); fig2b {stats}")
    v = fig3("d")
    if v: print("fig3 verdict:", v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
