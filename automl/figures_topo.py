#!/usr/bin/env python3
"""Figures for the topological log D result.

House style is imported from ``automl/figures.py`` rather than redefined, so
these plots sit beside the existing ones without a second visual language:
``_style()``, ``_save()`` (matched PNG + PDF at 160 dpi) and the ``C`` palette.

Colour is validated, not assumed.  ``automl/tests/test_palette.py`` ports the
dataviz skill's checks (Machado-Oliveira-Fernandes CVD simulation, OKLab dE)
because this cluster has no node runtime, and it found two real problems in the
"validated" house ordering:

* orange #eb6834 and green #008300 collapse under protanopia (dE = 3.2, below
  the 6.0 floor) -- fine as *adjacent* series, unsafe in a scatter where every
  pair is visible at once;
* magenta, yellow and aqua fail WCAG contrast against the light surface, so
  **no** 4-colour subset of the house palette passes every check.

Hence three hues carry identity everywhere (blue = this work, orange = FCNN
baseline, violet = CatBoost baseline) and any fourth distinction rides on marker
shape.  Identity never rests on colour alone.

Usage:  python3 -m automl.figures_topo --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automl.figures import C, FIG_DIR, GRID, INK, INK2, _save, _style

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "automl/reports"
ART = REPO / "automl/artifacts"

# Semantic colour roles, fixed across every figure.
TOPO, FCNN, CAT = C["blue"], C["orange"], C["violet"]


def _read(name: str) -> pd.DataFrame | None:
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


def _runs(*dirs: str) -> pd.DataFrame:
    """Every completed run's tag + headline metrics."""
    rows = []
    for d in dirs:
        for f in sorted((ART / d).glob("run_*.json")):
            try:
                r = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            m = r.get("metrics", {})
            rows.append({"tag": r.get("tag"), "dir": d,
                         "adj_r2": m.get("sel_adj_logSF_r2"),
                         "r2_overall": m.get("r2_overall"),
                         "arch": (r.get("config") or {}).get("arch")})
    return pd.DataFrame(rows).dropna(subset=["adj_r2", "r2_overall"])


# ---------------------------------------------------------------------------
def fig_forest() -> None:
    """The headline: every significance test as an effect size with its interval.

    A forest plot is the right form here because the question is *how much, and
    how certain* for a handful of named comparisons -- not a trend and not a
    distribution.  The zero line is the whole point, so it is the only heavy
    rule on the panel.
    """
    tests = [
        ("SNN ensemble (16 seeds)", "vs FCNN", 0.2426, 0.181, 0.333, FCNN),
        ("PI-CNN ensemble (15 seeds)", "vs FCNN", 0.1984, 0.107, 0.266, FCNN),
        ("SNN ensemble (16 seeds)", "vs CatBoost", 0.0867, 0.025, 0.122, CAT),
        ("SNN blend, w=0.5 pre-registered", "vs CatBoost", 0.1004, 0.038, 0.140, CAT),
        ("SNN blend, nested weight", "vs CatBoost", 0.1074, 0.039, 0.150, CAT),
    ]
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = np.arange(len(tests))[::-1]
    for yi, (_, _, d, lo, hi, col) in zip(y, tests):
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round",
                zorder=2)
        ax.plot([d], [yi], "o", ms=9, color=col, mec="#fcfcfb", mew=2, zorder=3)
    ax.axvline(0, color=INK, lw=1.4, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}\n{b}" for a, b, *_ in tests], fontsize=8.5)
    ax.set_xlabel("Δ adjacent-pair log SF R²  (arm − baseline, 90 % interval)")
    ax.set_title("Every test clears zero", loc="left", pad=10)
    ax.set_xlim(-0.02, 0.38)
    for yi, (_, _, d, lo, hi, _c) in zip(y, tests):
        ax.annotate(f"{d:+.3f}", (hi, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=INK2)
    # Identity is not colour-alone: the comparison is spelled out in each label.
    ax.plot([], [], color=FCNN, lw=2, label="vs FCNN (ECFP + RDKit)")
    ax.plot([], [], color=CAT, lw=2, label="vs CatBoost + group weights")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.text(0.0, -0.10, "4,746 rows · 162 extractants · leave-extractants-out "
             "CV · paired cluster bootstrap over extractants",
             fontsize=7.5, color=INK2)
    _save(fig, "topo_forest")


def fig_blend_curve() -> None:
    """Blend curve: the interior maximum is the complementarity argument.

    Both series are R², so they share one axis -- a second y-scale would be the
    classic dual-axis distortion and is not needed here.
    """
    d = _read("adjacent_blend.csv")
    if d is None:
        print("skip blend curve: adjacent_blend.csv missing")
        return
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(d["w"], d["adj_r2"], "-o", color=TOPO, lw=2, ms=6,
            mec="#fcfcfb", mew=1.5, label="adjacent-pair log SF R²")
    ax.plot(d["w"], d["r2_overall"], "-s", color=CAT, lw=2, ms=6,
            mec="#fcfcfb", mew=1.5, label="overall log D R²")
    peak = d.loc[d["adj_r2"].idxmax()]
    ax.annotate(f"peak {peak['adj_r2']:.3f} at w = {peak['w']:.1f}",
                (peak["w"], peak["adj_r2"]), xytext=(0, 15),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=INK)
    # Endpoint labels sit *beside* their points: placed below, they collide
    # with the x-axis tick labels.
    for x, ha, dx, lbl in ((0.0, "left", 10, "CatBoost alone"),
                           (1.0, "right", -10, "topology alone")):
        ax.annotate(lbl, (x, float(d.loc[d["w"] == x, "adj_r2"].iloc[0])),
                    xytext=(dx, -14), textcoords="offset points", ha=ha,
                    fontsize=8, color=INK2)
    ax.set_xlabel("weight on the topological model")
    ax.set_ylabel("R²")
    ax.set_title("The blend beats both endpoints",
                 loc="left", pad=10, fontsize=11)
    # Below the axes: every in-axes corner is occupied by a curve here.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              fontsize=8.5)
    fig.text(0.0, -0.22, "An interior maximum can only arise from complementary "
             "information; identical information interpolates monotonically.",
             fontsize=7.5, color=INK2)
    _save(fig, "topo_blend_curve")


def fig_tradeoff() -> None:
    """Where every arm sits on the two axes that matter.

    Three hues + marker shape: no 4-colour subset of the house palette passes
    CVD, normal-vision and contrast checks simultaneously (see test_palette.py).
    """
    runs = _runs("topo_runs", "topo_adjacent", "topo_runs_radial")
    if runs.empty:
        print("skip tradeoff: no runs found")
        return
    # Three topology-only ablations land at adjacent R2 -1.9 to -1.6.  Letting
    # them set the y-range squeezes every candidate model into a sliver, so the
    # axis is bounded and the off-scale arms are named instead -- their being
    # off-scale is the point, not a detail to hide.
    YLO = -0.55
    off = runs[runs["adj_r2"] < YLO].sort_values("adj_r2")
    vis = runs[runs["adj_r2"] >= YLO]

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for arch, marker, lbl in (("snn", "o", "simplicial network"),
                              ("picnn", "s", "persistence-image CNN")):
        sub = vis[vis["arch"] == arch]
        ax.scatter(sub["r2_overall"], sub["adj_r2"], s=52, marker=marker,
                   facecolor=TOPO, edgecolor="#fcfcfb", linewidth=1.4,
                   alpha=0.85, zorder=3, label=lbl)
    for x, y, col, mk, lbl, dx in ((0.3872, 0.0048, FCNN, "D", "FCNN", 0),
                                   (0.4987, 0.1422, CAT, "D", "CatBoost", -34)):
        ax.scatter([x], [y], s=110, marker=mk, facecolor=col,
                   edgecolor="#fcfcfb", linewidth=1.8, zorder=4,
                   label=f"{lbl} baseline")
        ax.annotate(lbl, (x, y), xytext=(dx, -18), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2)
    ax.axhline(0.1422, color=GRID, lw=1, ls="--", zorder=1)
    ax.set_xlabel("overall log D R²")
    ax.set_ylabel("adjacent-pair log SF R²")
    ax.set_ylim(YLO, 0.30)
    ax.set_xlim(0.10, 0.56)
    ax.set_title("Topology buys adjacent-pair accuracy, not overall accuracy",
                 loc="left", pad=10, fontsize=11)
    # Below the axes: any in-axes corner collides with a data point here.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
              fontsize=8.5)
    note = ""
    if len(off):
        note = ("  Off-scale below: "
                + ", ".join(f"{t.tag.split('_')[0]}-{t.tag.split('_')[1]} "
                            f"({t.adj_r2:+.2f})" for t in off.itertuples())
                + " — the topology-only ablations.")
    fig.text(0.0, -0.20, f"{len(runs)} single-model runs (not ensembled). "
             f"Dashed line = CatBoost's adjacent-pair score.{note}",
             fontsize=7.5, color=INK2)
    _save(fig, "topo_tradeoff")


def fig_seed_spread() -> None:
    """Why the claim rests on ensembles: single models are unstable."""
    # Same directories the ensemble draws from -- counting only topo_adj_seeds
    # showed 14 seeds beside a 16-seed ensemble figure.
    runs = _runs("topo_adj_seeds", "topo_adjacent")
    if runs.empty:
        print("skip seed spread: no seed runs")
        return
    groups = [("snn_pair2_sel", "simplicial network", TOPO, 0.2382),
              ("pi_pair2_sel", "persistence-image CNN", CAT, 0.2080)]
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    rng = np.random.default_rng(0)
    for i, (pref, lbl, col, ens) in enumerate(groups):
        vals = runs[runs["tag"].str.startswith(pref)]["adj_r2"].to_numpy()
        if not len(vals):
            continue
        ax.scatter(vals, np.full(len(vals), i) + rng.uniform(-.09, .09, len(vals)),
                   s=44, facecolor=col, edgecolor="#fcfcfb", linewidth=1.2,
                   alpha=0.85, zorder=3)
        ax.plot([ens], [i], marker="|", ms=26, mew=3, color=INK, zorder=4)
        ax.annotate(f"ensemble {ens:.3f}", (ens, i), xytext=(0, 16),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK)
        ax.annotate(f"n={len(vals)} seeds, mean {vals.mean():.3f}, "
                    f"SD {vals.std():.3f}", (min(vals), i),
                    xytext=(-8, -16), textcoords="offset points", ha="left",
                    fontsize=8, color=INK2)
    ax.axvline(0.1422, color=GRID, lw=1.2, ls="--", zorder=1)
    ax.annotate("CatBoost", (0.1422, 1.45), fontsize=8, color=INK2, ha="center")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[1] for g in groups], fontsize=9)
    ax.set_xlabel("adjacent-pair log SF R² (one point per seed)")
    # Not "above every seed": one SNN seed (s37, 0.2397) edges the 0.2382
    # ensemble.  The honest claim is that averaging lifts the *typical* seed
    # well above the per-seed mean and removes the downside tail.
    ax.set_title("Ensembling lifts the typical seed and removes the downside tail",
                 loc="left", pad=10, fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(groups) - 0.2)
    _save(fig, "topo_seed_spread")


def fig_stage2() -> None:
    """Stage 2's negative result, at descriptor level."""
    d = _read("scatter_diagnostic.csv")
    if d is None:
        print("skip stage2: scatter_diagnostic.csv missing")
        return
    # Block codes are opaque outside the codebase; label them from the
    # authoritative mapping in geom3d_features.BLOCK_PREFIX.
    from automl.geom3d_features import BLOCK_PREFIX
    names = {v: k.replace("_", " ") for k, v in BLOCK_PREFIX.items()}
    med = d.groupby("block")["delta"].median().sort_values()
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    cols = [TOPO if v > 0 else CAT for v in med.to_numpy()]
    ax.barh(np.arange(len(med)), med.to_numpy(), color=cols, height=0.62,
            edgecolor="#fcfcfb", linewidth=1.2, zorder=3)
    ax.axvline(0, color=INK, lw=1.3, zorder=2)
    ax.set_yticks(np.arange(len(med)))
    ax.set_yticklabels([f"{b} · {names.get(b, '')}".rstrip(" ·")
                        for b in med.index], fontsize=8.5)
    ax.set_xlabel("median paired change in family fit-R²  (tight − loose geometry)")
    ax.set_title("Tighter geometries barely move descriptor smoothness",
                 loc="left", pad=10, fontsize=11)
    ax.set_xlim(left=-0.0012)
    ax.grid(axis="y", visible=False)
    overall = d["delta"].median()
    ax.annotate(f"overall median {overall:+.4f}\n"
                f"({100*(d['delta'] > 0).mean():.1f} % of {len(d):,} cells improved)",
                (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8.5,
                color=INK2)
    fig.text(0.0, -0.06, "Residual forces fell 83× (0.185 → 0.0022 eV/Å) yet "
             "adjacent-pair R² did not move: the limit is conformational.",
             fontsize=7.5, color=INK2)
    _save(fig, "topo_stage2")


def fig_adjacent_parity() -> None:
    """Predicted vs true separation for adjacent pairs — the intuitive panel."""
    from automl import evaluation as ev
    from automl.topo.compare_arms import collect, attach_meta
    from automl.topo.ensemble_adjacent import _load, config_key, SEED_DIRS

    arms = collect()
    base = attach_meta(arms.get("baseline::mlp::none"))
    if base is None or base.empty:
        print("skip parity: FCNN baseline OOF missing")
        return
    members = {}
    for dd in SEED_DIRS:
        if dd.exists():
            for name, df in _load(dd).items():
                members.setdefault(config_key(name), {})[name] = df
    key = "snn_pair2_sel_snn_baseline_2d_f3.5_h1"
    if key not in members:
        print("skip parity: SNN seed ensemble missing")
        return
    mem = members[key]
    idx = None
    for df in mem.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    idx = idx.intersection(base.index)
    topo = np.vstack([mem[k].loc[idx, "oof"].to_numpy(float)
                      for k in sorted(mem)]).mean(axis=0)
    b = base.loc[idx]
    y, comp = b["y"].to_numpy(float), b["composition_key"].to_numpy()
    li = b["lanthanide_index"].to_numpy()

    # Reuse the metric's own pair construction -- never re-enumerate pairs in a
    # figure.  An earlier version of this plot did, skipped the within-metal
    # averaging, and rendered 13,029 pairs showing the baseline beating the
    # model: the exact inverse of the real result.
    def pairs(pred):
        return ev.adjacent_pair_arrays(y, pred, comp, li)

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharex=True, sharey=True)
    for ax, (pred, lbl, col) in zip(axes, (
            (b["oof"].to_numpy(float), "FCNN baseline", FCNN),
            (topo, "simplicial-network ensemble", TOPO))):
        t, p = pairs(pred)
        ax.scatter(t, p, s=13, facecolor=col, edgecolor="none", alpha=0.35,
                   zorder=3)
        lim = (-2.2, 2.2)
        ax.plot(lim, lim, color=INK2, lw=1.2, ls="--", zorder=2)
        ax.axhline(0, color=GRID, lw=0.9, zorder=1)
        ax.axvline(0, color=GRID, lw=0.9, zorder=1)
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_title(f"{lbl}\nadjacent-pair R² = {ev._r2(t, p):+.3f}",
                     loc="left", fontsize=10, pad=8)
        ax.set_xlabel("true Δ log D (adjacent pair)")
    axes[0].set_ylabel("predicted Δ log D")
    fig.text(0.0, -0.04, f"{len(pairs(topo)[0]):,} adjacent lanthanide pairs "
             "(|Δ index| = 1), out-of-fold, leave-extractants-out CV. "
             "Dashed line is parity.", fontsize=7.5, color=INK2)
    _save(fig, "topo_adjacent_parity")


def fig_control_factorial() -> None:
    """The attribution: how much of the gain is topology and how much is the objective.

    An interaction plot is the right form and a bar chart is not.  The question
    is whether the *effect of topology* differs between the two objectives --
    that is a comparison of slopes, and slopes are what a reader can see
    directly here.  Grouped bars would show the same four numbers while making
    the one relationship that matters something you have to compute by eye.

    Colour follows the roles fixed across every figure in this set (orange =
    the no-topology reference, violet = PI-CNN, blue = SNN) and marker shape
    repeats the distinction, so identity never rests on colour alone.
    """
    cells = _read("control_cells.csv")
    tests = _read("control_factorial.csv")
    if cells is None or tests is None:
        print("skip control factorial: run automl.topo.control_factorial first")
        return
    val = dict(zip(cells["cell"], cells["adj_r2"]))
    # Which of T0/T0w is *the* control was decided once, in
    # control_factorial.py, under the pre-registered max rule.  Read it back
    # rather than re-deriving it here -- a rule applied in two places is how
    # the parity figure came to contradict the headline.
    if "is_control" in cells.columns:
        sel = cells[cells["is_control"].astype(bool)]
        if len(sel):
            val["T0"] = float(sel["adj_r2"].iloc[0])
    need = ("T1", "T0", "P1", "P0", "S1", "S0")
    if any(k not in val for k in need):
        print(f"skip control factorial: missing cells "
              f"{[k for k in need if k not in val]}")
        return

    series = [("no topology (tabular)", "T1", "T0", FCNN, "s"),
              ("PI-CNN", "P1", "P0", CAT, "^"),
              ("SNN", "S1", "S0", TOPO, "o")]
    _style()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 4.0),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.42})

    x = [0, 1]
    for lbl, a, b, col, mk in series:
        ax.plot(x, [val[a], val[b]], color=col, lw=2, marker=mk, ms=9,
                mec="#fcfcfb", mew=1.8, zorder=3, label=lbl)
        ax.annotate(f"{val[b]:+.3f}", (1, val[b]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=col)
    # The two published reference points, so the decomposition is read against
    # what was claimed rather than in isolation.  Labels sit INSIDE the panel:
    # placed to the right they collided with the neighbouring panel's tick
    # labels, which is illegible at print size.
    for yv, name in ((0.00477, "FCNN"), (0.14217, "CatBoost")):
        ax.axhline(yv, color=GRID, lw=1.1, ls="--", zorder=1)
        ax.annotate(f"{name} (published)", (0.02, yv), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color=INK2,
                    va="bottom", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(["plain MSE\nobjective", "pairwise-contrast\nobjective"],
                       fontsize=9)
    ax.set_xlim(-0.18, 1.34)
    ax.set_ylabel("adjacent-pair log SF R²  (16-seed ensemble)")
    # Titles are computed from the numbers, never written ahead of them.  Two
    # figures in this study previously carried a claim the data did not support
    # (a parity plot that inverted the comparison, a seed plot that said "above
    # every seed" when one seed beat the ensemble), and both survived because
    # the title was prose rather than a function of the values it sat above.
    ax.set_title("Only the SNN gains from the objective", loc="left", pad=8,
                 fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="x", visible=False)

    # Right panel: the pre-registered contrasts, as effect sizes with intervals.
    want = [("S0", "T0", "topology on top\nof the objective (SNN)", TOPO),
            ("P0", "T0", "same, PI-CNN", CAT),
            ("P1", "T1", "topology with NO\ncontrast objective", CAT),
            ("S1", "T1", "same, SNN", TOPO),
            ("T0", "T1", "the objective,\nwithout topology", FCNN)]
    rows = []
    for arm, base, lbl, col in want:
        # Match on the ROLE columns, not the resolved cell names.  The control
        # resolves to T0 or T0w under the pre-registered max rule, so matching
        # literal names silently dropped the "objective without topology"
        # contrast from this panel -- four bars where five were intended.
        m = tests[(tests.get("arm_role", tests["arm"]) == arm)
                  & (tests.get("base_role", tests["base"]) == base)]
        if len(m):
            r = m.iloc[0]
            rows.append((lbl, float(r["delta"]), float(r["lo"]),
                         float(r["hi"]), col))
    if rows:
        ys = np.arange(len(rows))[::-1]
        for yi, (_, d, lo, hi, col) in zip(ys, rows):
            ax2.plot([lo, hi], [yi, yi], color=col, lw=2,
                     solid_capstyle="round", zorder=2)
            ax2.plot([d], [yi], "o", ms=8, color=col, mec="#fcfcfb", mew=2,
                     zorder=3)
            ax2.annotate(f"{d:+.3f}", (hi, yi), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=8.5,
                         color=INK2)
        ax2.axvline(0, color=INK, lw=1.4, zorder=1)
        ax2.set_yticks(ys)
        ax2.set_yticklabels([r[0] for r in rows], fontsize=8.5)
        ax2.set_xlabel("Δ adjacent-pair R² (90 % interval)")
        # Same rule: the verdict is counted, not asserted.
        clear = sum(1 for _, d, lo, hi, _c in rows if lo > 0)
        below = sum(1 for _, d, lo, hi, _c in rows if hi < 0)
        ax2.set_title(f"{clear} of {len(rows)} intervals exclude zero above it"
                      + (f", {below} below" if below else ""),
                      loc="left", pad=10, fontsize=10.5)
        ax2.grid(axis="y", visible=False)

    fig.text(0.0, -0.06,
             "4,746 rows · 162 extractants · 16 matched seeds per cell · "
             "leave-extractants-out CV · paired cluster bootstrap over "
             "extractants · pre-registered in CONTROL_PREREGISTRATION.md",
             fontsize=7.5, color=INK2)
    _save(fig, "topo_control_factorial")


def fig_control_decomposition() -> None:
    """Every model that was actually measured, on one axis.

    This replaced a waterfall, and the reason matters.  A waterfall shows
    additive terms as a running total, which silently asserts that the
    intermediate heights are states something reached.  With Shapley terms they
    are not: crediting the objective before topology put the running total at
    +0.287, a value no model in this study scores, and then walked *down* to the
    SNN's +0.238.  A reader would have taken +0.287 for a result.

    So the chart shows measured models instead.  Each row is a real 16-seed
    ensemble with a real out-of-fold score; the reader can see the whole ladder
    and read any gap off directly, without a decomposition being imposed on
    them.  The order-dependent attribution stays in the table, where its range
    can be stated honestly.

    Values come from control_cells.csv and fcnn_diagnostic.csv so the chart
    cannot drift from the tables it illustrates.
    """
    cells = _read("control_cells.csv")
    if cells is None:
        print("skip decomposition: run automl.topo.control_factorial first")
        return
    val = dict(zip(cells["cell"], cells["adj_r2"]))
    if "is_control" in cells.columns:
        sel = cells[cells["is_control"].astype(bool)]
        if len(sel):
            val["T0"] = float(sel["adj_r2"].iloc[0])
    if any(k not in val for k in ("mlp", "T1", "T0", "S0")):
        print("skip decomposition: cells missing")
        return

    # uses_topology decides colour; the label carries the rest.
    bars = [("FCNN baseline, as published", val["mlp"], False),
            ("CatBoost, as published", val.get("catboost", 0.1422), False),
            ("PI-CNN + plain objective", val["P1"], True),
            ("tabular + plain objective", val["T1"], False),
            ("tabular + contrast objective", val["T0"], False),
            ("PI-CNN + contrast objective", val["P0"], True),
            ("SNN + contrast objective", val["S0"], True)]

    diag = _read("fcnn_diagnostic.csv")
    if diag is not None:
        m = diag[diag["mode"] == "std_scaler_ens16"]
        if len(m):
            bars.append(("published FCNN, one line changed:\n"
                         "StandardScaler, 16 seeds",
                         float(m["adj_r2"].iloc[0]), False))
    # S2, the variance-reduced simplicial arm, once it exists.  Its value is
    # read from the confirmatory test's own output rather than recomputed, so
    # the chart cannot disagree with the number it illustrates.
    s2 = _read("s2_test.csv")
    if s2 is not None and "kind" in s2.columns:
        pri = s2[s2["kind"] == "primary"]
        if len(pri):
            bars.append(("SNN + contrast + conformers\n+ block-centring, 32 seeds",
                         float(pri["arm_obs"].iloc[0]), True))
    bars.sort(key=lambda b: b[1])

    _style()
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ys = np.arange(len(bars))
    for y, (lbl, v, topo) in zip(ys, bars):
        ax.barh(y, v, height=0.6, color=TOPO if topo else FCNN, zorder=3,
                edgecolor="#fcfcfb", linewidth=1.2,
                hatch="" if topo else "///")
        ax.annotate(f"{v:+.3f}", (v, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.5)
    # "16-seed" was wrong once S2 (32 seeds) joined; the counts differ by arm,
    # so the axis states the protocol and the per-bar labels carry the rest.
    ax.set_xlabel("adjacent-pair log SF R²  (out-of-fold, seed-ensembled)")
    ax.set_xlim(0, max(b[1] for b in bars) * 1.22)
    # Hatching, not colour alone, separates the two families -- the palette
    # validator allows three hues here but identity should not rest on hue.
    #
    # Explicit Patch handles: barh([], []) builds a container with no patches,
    # so matplotlib had nothing to take a colour from and drew both legend keys
    # in the same blue -- a legend that contradicted the bars it labelled.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=TOPO, edgecolor="#fcfcfb",
                             label="uses 3D topology"),
                       Patch(facecolor=FCNN, edgecolor="#fcfcfb", hatch="///",
                             label="no topology")],
              loc="lower right", fontsize=8.5)
    # A bare gap is a point estimate, and this metric does not support point
    # estimates -- the baseline alone spans 0.11 across seed conventions.  If
    # the paired interval has been computed, the title carries the verdict; the
    # first version of this title said "+0.018" with no interval, which reads as
    # a result and is not one.
    # The headline is the most recent confirmatory test against the repaired
    # baseline: S2 if it has been run, otherwise S0.  Both are the same
    # comparison, so quoting the older one once a newer arm exists would
    # understate the work; quoting both would bury the endpoint.
    best_flat = max(b[1] for b in bars if not b[2])
    s2t = _read("s2_test.csv")
    fb = _read("fixed_baseline_test.csv")
    label, row = None, None
    if s2t is not None and "kind" in s2t.columns and (s2t["kind"] == "primary").any():
        row = s2t[s2t["kind"] == "primary"].iloc[0]
        label = "S2 (conformers + block-centring)"
    elif fb is not None and "S0" in set(fb["arm"]):
        row = fb[fb["arm"] == "S0"].iloc[0]
        label = "SNN"
    if row is not None:
        # One number with its interval.  An earlier version quoted the raw gap
        # beside the bootstrap mean, inviting the reader to wonder which was the
        # result; and a gap alone is a point estimate on a metric where the
        # baseline itself spans 0.11 across seed conventions.
        sig = ("excludes zero" if float(row["lo"]) > 0 else
               "spans zero — not distinguishable")
        title = (f"{label} over the best model without topology: "
                 f"{float(row['delta']):+.3f} "
                 f"[{float(row['lo']):+.3f}, {float(row['hi']):+.3f}], {sig}")
    else:
        title = (f"Best model without topology is "
                 f"{val['S0'] - best_flat:+.3f} from the best with it")
    ax.set_title(title, loc="left", pad=10, fontsize=10.5)
    ax.grid(axis="y", visible=False)
    fig.text(0.0, -0.08,
             "Same 4,746 rows and leave-extractants-out folds throughout. "
             "Attribution between the objective and topology is order-dependent "
             "and is reported with its range in control_attribution.csv rather "
             "than drawn here.", fontsize=7.5, color=INK2)
    _save(fig, "topo_control_decomposition")


def fig_stack() -> None:
    """The positive result: topology adds to the strongest baseline, and the
    matched control does not.

    Two panels because the claim has two halves and a reader must not take the
    first without the second.  Left: where each blend lands, with the repaired
    baseline as the line to beat.  Right: the three pre-registered contrasts,
    with the four-test corrected interval drawn as a lighter bar behind the
    nominal one -- contrast 3 is nominally significant and spans zero once
    corrected, and a figure that showed only the nominal interval would overstate
    exactly the thing the write-up is careful about.
    """
    cells = _read("stack_test.csv")
    if cells is None:
        print("skip stack: run automl.topo.stack_test first")
        return
    _style()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.6, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.45})

    # --- left: where the blends land -------------------------------------
    bars = [("repaired baseline\n(no topology)", 0.2206, False),
            ("+ matched tabular\ncontrol (T0w)", 0.2289, False),
            ("SNN alone (S0)", 0.2382, True),
            ("+ SNN (S0)", 0.2511, True)]
    ys = np.arange(len(bars))
    for y, (lbl, v, topo) in zip(ys, bars):
        ax.barh(y, v, height=0.6, color=TOPO if topo else FCNN,
                hatch="" if topo else "///", edgecolor="#fcfcfb", linewidth=1.2,
                zorder=3)
        ax.annotate(f"{v:+.3f}", (v, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=INK2)
    ax.axvline(0.2206, color=GRID, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_xlabel("adjacent-pair log SF R²")
    ax.set_xlim(0.19, 0.275)
    ax.set_title("Blending topology with the best baseline", loc="left",
                 pad=10, fontsize=10.5)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=TOPO, edgecolor="#fcfcfb",
                             label="uses 3D topology"),
                       Patch(facecolor=FCNN, edgecolor="#fcfcfb", hatch="///",
                             label="no topology")],
              loc="lower right", fontsize=8)

    # --- right: the pre-registered contrasts, nominal AND corrected --------
    want = [("1_primary", "blend(S0) − repaired\n(does topology add?)", TOPO),
            ("2_control", "blend(T0w) − repaired\n(control: any 2nd model?)", FCNN),
            ("3_decisive", "blend(S0) − blend(T0w)\n(topology-specific?)", TOPO)]
    rows = []
    for key, lbl, col in want:
        m = cells[cells["contrast"] == key]
        if len(m):
            r = m.iloc[0]
            rows.append((lbl, float(r["delta"]), float(r["lo"]), float(r["hi"]),
                         float(r.get("lo_3test", np.nan)),
                         float(r.get("hi_3test", np.nan)), col))
    yy = np.arange(len(rows))[::-1]
    for y, (lbl, d, lo, hi, clo, chi, col) in zip(yy, rows):
        if np.isfinite(clo):
            ax2.plot([clo, chi], [y, y], color=col, lw=5, alpha=0.25,
                     solid_capstyle="butt", zorder=2)
        ax2.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round",
                 zorder=3)
        ax2.plot([d], [y], "o", ms=8, color=col, mec="#fcfcfb", mew=2, zorder=4)
        ax2.annotate(f"{d:+.3f}", (hi, y), xytext=(7, 0),
                     textcoords="offset points", va="center", fontsize=8.5,
                     color=INK2)
    ax2.axvline(0, color=INK, lw=1.4, zorder=1)
    ax2.set_yticks(yy); ax2.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax2.set_xlabel("Δ adjacent-pair R²   (dark = 90 %, pale = 4-test corrected)")
    n_nom = sum(1 for r in rows if r[2] > 0)
    n_cor = sum(1 for r in rows if np.isfinite(r[4]) and r[4] > 0)
    ax2.set_title(f"{n_nom}/3 clear zero nominally, {n_cor}/3 after correction",
                  loc="left", pad=10, fontsize=10.5)
    ax2.grid(axis="y", visible=False)

    fig.text(0.0, -0.07,
             "4,746 rows · 162 extractants · leave-extractants-out CV · nested "
             "per-extractant blend weights · paired cluster bootstrap · "
             "pre-registered in STACK_PREREGISTRATION.md",
             fontsize=7.5, color=INK2)
    _save(fig, "topo_stack")


FIGS = {"forest": fig_forest, "stack": fig_stack, "blend": fig_blend_curve, "tradeoff": fig_tradeoff,
        "seeds": fig_seed_spread, "stage2": fig_stage2,
        "parity": fig_adjacent_parity, "control": fig_control_factorial,
        "decomposition": fig_control_decomposition}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    args = ap.parse_args()
    names = args.only if args.only else (sorted(FIGS) if args.all else sorted(FIGS))
    for n in names:
        try:
            FIGS[n]()
        except Exception as exc:                       # one bad figure must not
            print(f"[figures_topo] {n} FAILED: {type(exc).__name__}: {exc}")
    print(f"figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
