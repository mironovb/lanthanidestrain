#!/usr/bin/env python3
"""The three figures that go with `reports/PI_EMAIL_2026-08-03.md`.

One figure per claim the email actually makes, in the order it makes them:

1. ``email_fig1_result``  -- the levels: what each model scores, and what the
   deployable combination reaches with and without the 3D arm.
2. ``email_fig2_control`` -- the three pre-registered contrasts, including the
   matched control whose null is what makes this a structural result rather
   than an ensembling one.
3. ``email_fig3_limits``  -- the two qualifications the email carries: the
   effect is not topological, and its size depends on how strictly "identical
   conditions" is defined.

House style is imported from ``automl/figures.py`` rather than redefined, so
these sit beside the existing plots without a second visual language.

Every number is read from the result CSVs and every title is an f-string over
the data it describes.  This project has shipped a figure and a table computing
the same quantity two different ways before; nothing here is retyped.

Usage:  module load anaconda/Python-ML-2025a
        PYTHONPATH=$PWD python3 -m automl.figures_pi_email --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automl.figures import C, GRID, INK, INK2, _save, _style

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "automl/reports"

TOPO, FCNN, CAT = C["blue"], C["orange"], C["violet"]
WARN = C["red"]


def _read(name: str) -> pd.DataFrame | None:
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


def _need(*names: str) -> list[pd.DataFrame] | None:
    """Read every CSV a figure needs, or report which one is missing."""
    out = []
    for n in names:
        df = _read(n)
        if df is None or df.empty:
            print(f"skip: {n} absent or empty")
            return None
        out.append(df)
    return out


def _ci(ax, y, row, lo_c: str, hi_c: str, colour: str) -> None:
    """One forest row: thick 90 % interval, thin multiplicity-corrected one."""
    ax.plot([row["lo"], row["hi"]], [y, y], color=colour, lw=3.4,
            solid_capstyle="round", zorder=3)
    ax.plot([row[lo_c], row[hi_c]], [y, y], color=colour, lw=1.1, alpha=0.75,
            zorder=2)
    ax.plot([row["delta"]], [y], "o", color=colour, ms=7,
            markeredgecolor="white", markeredgewidth=0.9, zorder=4)


# ---------------------------------------------------------------------------
def fig_result() -> None:
    """The levels behind "0.267 against 0.226".

    Bar heights are levels only.  No difference is read off this figure: the
    paired bootstrap contrasts are computed on shared pairs and do not equal
    the difference of two whole-set scores, so quoting one here would disagree
    with `email_fig2_control` by about 0.003.  Contrasts live in that figure.
    """
    got = _need("dualkey_arms.csv", "best_stack.csv", "stack_test.csv")
    if got is None:
        return
    arms, stack, two = got
    a = arms.set_index("arm")["adj_r2_binned"]

    drop = stack[stack["base"] == "no topology (CatBoost+repaired)"].iloc[0]
    swap = stack[stack["base"] == "topology swapped for control"].iloc[0]
    full = float(drop["arm_obs"])
    no3d = float(drop["baseline_obs"])
    ctrl = float(swap["baseline_obs"])
    blend = float(two.set_index("contrast").loc["1_primary", "arm_obs"])

    rows = [
        ("CatBoost alone", float(a["CatBoost"]), CAT, 0.5),
        ("fingerprint network alone", float(a["repaired"]), FCNN, 0.5),
        ("3D encoder alone", float(a["S0"]), TOPO, 0.5),
        ("", np.nan, None, 0.0),
        ("fingerprint network + 3D encoder", blend, TOPO, 0.95),
        ("", np.nan, None, 0.0),
        ("CatBoost + fingerprint network", no3d, INK2, 0.9),
        ("      + a matched model with no 3D input", ctrl, WARN, 0.9),
        ("      + the 3D encoder", full, TOPO, 0.95),
    ]

    _style()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ticks, labels = [], []
    for i, (label, v, colour, alpha) in enumerate(rows):
        y = -i
        if not np.isfinite(v):
            continue
        ax.barh(y, v, color=colour, alpha=alpha, height=0.62)
        ax.text(v + 0.004, y, f"{v:+.3f}", va="center", fontsize=9, color=INK2)
        ticks.append(y)
        labels.append(label)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_ylim(-len(rows) + 0.4, 0.6)
    ax.set_xlim(0, full * 1.18)
    ax.set_xlabel("adjacent-lanthanide separation R²   (0 = no better than the "
                  "average separation)")
    ax.set_title(
        f"Every combination containing the 3D encoder scores highest\n"
        f"the deployable three-model combination reaches {full:.3f}, against "
        f"{no3d:.3f} without the 3D arm\nand {ctrl:.3f} when that slot holds a "
        f"model with no 3D input", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    _save(fig, "email_fig1_result")


# ---------------------------------------------------------------------------
def fig_control() -> None:
    """The three pre-registered contrasts, control included."""
    got = _need("stack_test.csv")
    if got is None:
        return
    df = got[0].set_index("contrast")
    lo_c = [c for c in df.columns if c.startswith("lo_") and c != "lo"][0]
    hi_c = [c for c in df.columns if c.startswith("hi_") and c != "hi"][0]
    looks = lo_c.split("_")[1]

    rows = [
        ("1_primary", "add the 3D encoder to the\nfingerprint network", TOPO),
        ("2_control", "add a matched model with\nno 3D input instead", WARN),
        ("3_decisive", "3D encoder against that\ncontrol, in the same slot", TOPO),
    ]

    _style()
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for i, (key, label, colour) in enumerate(rows):
        r = df.loc[key]
        _ci(ax, -i, r, lo_c, hi_c, colour)
        ax.text(float(r[hi_c]) + 0.003, -i, f"{float(r['delta']):+.4f}",
                va="center", fontsize=9, color=colour)
        if str(r[f"verdict_{looks}"]).startswith("not") and float(r["lo"]) > 0:
            ax.text(float(r["lo"]) - 0.002, -i - 0.26,
                    "corrected for multiple looks, this one spans zero",
                    ha="right", va="top", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    ax.set_ylim(-len(rows) + 0.5, 0.5)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([r[1] for r in rows], fontsize=9.5)
    ax.set_xlabel(f"Δ adjacent-pair R²   (thick = 90 % interval, "
                  f"thin = corrected for {looks.replace('test', ' tests')})")

    prim = float(df.loc["1_primary", "delta"])
    ctrl = float(df.loc["2_control", "delta"])
    ax.set_title(
        f"A second model does not help on its own; the 3D one does\n"
        f"matched control {ctrl:+.4f}, interval spans zero  ·  "
        f"3D encoder {prim:+.4f}, interval excludes zero",
        loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.13)
    _save(fig, "email_fig2_control")


# ---------------------------------------------------------------------------
def fig_limits() -> None:
    """Left: it is not topological.  Right: it depends on the blocking."""
    got = _need("encoder_test.csv", "encoder_arms.csv", "dualkey_test.csv")
    if got is None:
        return
    enc, arms, dual = got
    enc = enc[enc["key"] == "composition_key"]
    a = arms.set_index("arm")["adj_r2_binned"]
    lo_e = [c for c in enc.columns if c.startswith("lo_") and c != "lo"][0]
    hi_e = [c for c in enc.columns if c.startswith("hi_") and c != "hi"][0]

    _style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 3.9),
                                  gridspec_kw={"width_ratios": [1.05, 1],
                                               "wspace": 0.42})

    # -- left: three 3D encoders, one of them with no topology in it ---------
    enc_rows = [
        ("with G0", "no topology",
         f"graph over the same complex,\ntriangles deleted  ({a['G0']:+.3f})",
         TOPO),
        ("with D0", "no topology",
         f"distance network,\nno simplices at all  ({a['D0']:+.3f})", TOPO),
        ("with S0", "with D0",
         f"simplicial against distance,\nsame slot  ({a['S0']:+.3f} vs "
         f"{a['D0']:+.3f})", INK2),
    ]
    labels = []
    for i, (arm, base, label, colour) in enumerate(enc_rows):
        r = enc[(enc["arm"] == arm) & (enc["base"] == base)]
        if r.empty:
            continue
        r = r.iloc[0]
        _ci(ax, -i, r, lo_e, hi_e, colour)
        ax.text(float(r[hi_e]) + 0.002, -i, f"{float(r['delta']):+.4f}",
                va="center", fontsize=9, color=colour)
        labels.append(label)
    ax.axvline(0, color=INK, lw=1.0, zorder=1)
    ax.set_yticks([-i for i in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-len(labels) + 0.5, 0.5)
    ax.set_xlabel("Δ adjacent-pair R²")
    dec = enc[(enc["arm"] == "with S0") & (enc["base"] == "with D0")].iloc[0]
    ax.set_title(
        f"It is 3D, not topology\n"
        f"a network with no simplices earns the same slot,\n"
        f"and the two differ by {float(dec['delta']):+.4f}",
        loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.16)

    # -- right: the same contrasts under both block keys ---------------------
    lo_d = [c for c in dual.columns if c.startswith("lo_") and c != "lo"][0]
    hi_d = [c for c in dual.columns if c.startswith("hi_") and c != "hi"][0]
    q_drop = "drop-in: does adding S0 to the best no-topology stack help?"
    q_swap = "swap: S0 vs the matched tabular control in the same slot"
    keys = [("composition_key", TOPO, "binned conditions (published)"),
            ("strict_composition_key", WARN, "every condition matched")]
    yticks, ylabels = [], []
    for i, (q, name) in enumerate([(q_drop, "3D encoder added to the\nbest "
                                            "combination without it"),
                                   (q_swap, "3D encoder against the\nmatched "
                                            "control")]):
        for j, (key, colour, legend) in enumerate(keys):
            r = dual[(dual["question"] == q) & (dual["key"] == key)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = -i + (0.18 if j == 0 else -0.18)
            _ci(ax2, y, r, lo_d, hi_d, colour)
            ax2.text(float(r[hi_d]) + 0.002, y, f"{float(r['delta']):+.4f}",
                     va="center", fontsize=8.5, color=colour)
            if i == 0:
                ax2.plot([], [], color=colour, lw=3.4, label=legend)
        yticks.append(-i)
        ylabels.append(name)
    ax2.axvline(0, color=INK, lw=1.0, zorder=1)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels, fontsize=8.5)
    ax2.set_ylim(-len(yticks) + 0.4, 0.6)
    ax2.set_xlabel("Δ adjacent-pair R²")
    b = float(dual[(dual["question"] == q_drop) &
                   (dual["key"] == "composition_key")]["delta"].iloc[0])
    s = float(dual[(dual["question"] == q_drop) &
                   (dual["key"] == "strict_composition_key")]["delta"].iloc[0])
    p = float(dual[(dual["question"] == q_drop) &
                   (dual["key"] == "strict_composition_key")]["p_better"].iloc[0])
    ax2.set_title(
        f"Matching conditions strictly halves it\n"
        f"{b:+.4f} binned against {s:+.4f} strict,\n"
        f"still positive at P = {p:.2f} but no longer clear of zero",
        loc="left", fontsize=11)
    ax2.grid(axis="y", visible=False)
    ax2.margins(x=0.16)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2,
               fontsize=8.5, columnspacing=1.2, handletextpad=0.5)
    _save(fig, "email_fig3_limits")


FIGS = {"result": fig_result, "control": fig_control, "limits": fig_limits}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    args = ap.parse_args()
    want = args.only if args.only else (sorted(FIGS) if args.all else [])
    if not want:
        ap.error("choose --all or --only NAME")
    for n in want:
        FIGS[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
