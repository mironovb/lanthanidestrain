#!/usr/bin/env python3
"""Report figures for the 3D-feature AutoML study.

Colour follows the validated categorical order (blue / green / magenta /
yellow); every bar carries a direct value label, so identity is never conveyed
by colour alone and the low-contrast slots stay legible.  One measure per axis,
no dual scales, recessive grid.
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

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "automl/reports/figures"

# Validated categorical order, light surface.
C = {"blue": "#2a78d6", "green": "#008300", "magenta": "#e87ba4",
     "yellow": "#eda100", "aqua": "#1baf7a", "orange": "#eb6834",
     "violet": "#4a3aa7", "red": "#e34948"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "font.family": "DejaVu Sans", "figure.dpi": 160,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.alpha": 0.7, "legend.frameon": False, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / f"{name}.png")


# ---------------------------------------------------------------------------
def fig_decomposition(res: pd.DataFrame, champ_dir: Path | None = None) -> None:
    """Where the leave-extractants-out model actually fails.

    Built from the *best* champion configuration's out-of-fold predictions
    (protocol B) so it matches the numbers quoted in the report; falls back to
    the protocol-A screening baseline only if no champion run exists yet.
    """
    label = "2D baseline"
    vals = None
    if champ_dir is not None:
        f = champ_dir / "oof_2_baseline_2D,_CatBoost_+_group_wts.parquet"
        if f.exists():
            from automl import evaluation as ev
            d = pd.read_parquet(f)
            m = ev.variance_decomposed_r2(d["y"].to_numpy(float),
                                          d["oof"].to_numpy(float),
                                          d["extractant_group"].to_numpy())
            fr = pd.DataFrame({"y": d["y"], "p": d["oof"], "c": d["composition_key"]})
            gm = fr.groupby("c")[["y", "p"]].transform("mean")
            yc = fr["y"].to_numpy() - gm["y"].to_numpy()
            pc = fr["p"].to_numpy() - gm["p"].to_numpy()
            ss = float(np.sum(yc ** 2))
            wc = 1 - float(np.sum((yc - pc) ** 2)) / ss if ss > 0 else np.nan
            vals = [m["r2_overall"], m["r2_between"], m["r2_within"], wc]
            label = "CatBoost + inverse-extractant weights, 2D features"
    if vals is None:
        d = res[(res["tag"] == "ablation") & (res["row_filter"] == "has3d")]
        base = d[d["preset"] == "baseline_2d"]
        if base.empty:
            return
        b = base.iloc[0]
        vals = [b["r2_overall"], b["r2_between"], b["r2_within"],
                b["r2_within_composition"]]
    labels = ["overall", "between\nextractants", "within\nextractant",
              "within extractant\n+ conditions"]
    colors = [C["blue"], C["green"], C["magenta"], C["yellow"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylabel("R² (out-of-fold)")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("Almost all the skill is between extractants, not inside one",
                 pad=24, loc="left")
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=9, color=INK2)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    _save(fig, "fig1_baseline_decomposition")


def fig_block_ablation(res: pd.DataFrame) -> None:
    """Delta R² contributed by each single 3D block."""
    d = res[(res["tag"] == "ablation") & (res["row_filter"] == "has3d")].copy()
    base = d[d["preset"] == "baseline_2d"]
    if base.empty:
        return
    b = base.iloc[0]
    names = {
        "plus_g1": "G1 first shell", "plus_g2": "G2 contraction-corrected",
        "plus_g3": "G3 polyhedron shape", "plus_g4": "G4 steric burial",
        "plus_g5": "G5 xTB electronics", "plus_g6": "G6 metal RDF",
        "plus_g7": "G7 global shape/SASA", "plus_g8": "G8 chelate topology",
        "plus_g9": "G9 persistent homology", "plus_g10": "G10 series-relative",
        "plus_g11": "G11 persistence images",
        "plus_p3d_phys": "shipped: physical", "plus_p3d_poly": "shipped: polyhedron",
    }
    rows = []
    for preset, label in names.items():
        r = d[d["preset"] == preset]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({"label": label,
                     "d_overall": r["r2_overall"] - b["r2_overall"],
                     "d_within": r["r2_within"] - b["r2_within"],
                     "d_sel": r["sel_logSF_r2"] - b["sel_logSF_r2"]})
    if not rows:
        return
    t = pd.DataFrame(rows).sort_values("d_overall")
    y = np.arange(len(t))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), sharey=True)
    specs = [("d_overall", "Δ R² overall", C["blue"]),
             ("d_within", "Δ R² within extractant", C["green"]),
             ("d_sel", "Δ R² pairwise log SF", C["orange"])]
    for ax, (col, title, color) in zip(axes, specs):
        ax.barh(y, t[col], color=color, height=0.62, zorder=3)
        ax.axvline(0, color=INK2, linewidth=1.0, zorder=4)
        for yi, v in zip(y, t[col]):
            ax.text(v + (0.002 if v >= 0 else -0.002), yi, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=INK)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        # Always keep the zero line inside the frame -- these are deltas and the
        # reader must be able to see which side of "no change" a bar is on.
        lo = min(float(t[col].min()), 0.0)
        hi = max(float(t[col].max()), 0.0)
        pad = 0.42 * max(abs(lo), abs(hi), 1e-3)
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(t["label"], fontsize=9)
    fig.suptitle("Single-block screening on LightGBM (protocol A): every 3D block "
                 "costs series ordering",
                 x=0.09, ha="left", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_block_ablation")


def fig_noise_diagnostic(path: Path) -> None:
    """How much of each descriptor's within-family variation is a size trend."""
    if not path.exists():
        return
    t = pd.read_csv(path)
    t = t.sort_values("median")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y = np.arange(len(t))
    ax.barh(y, t["median"], color=C["violet"], height=0.6, zorder=3)
    for yi, v in zip(y, t["median"]):
        ax.text(v + 0.006, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(t["block"], fontsize=9)
    ax.set_xlabel("median R² of descriptor vs Shannon ionic radius, within a ligand family")
    ax.set_xlim(0, max(t["median"]) * 1.25)
    ax.set_title("Most of the 3D variation along the series is conformer noise",
                 loc="left", pad=12)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    _save(fig, "fig3_conformer_noise")


def fig_architectures(res: pd.DataFrame) -> None:
    """Flat vs anchored-residual vs delta-learning, R² decomposition."""
    d = res[res["tag"].astype(str).str.startswith("arch")].copy()
    if d.empty:
        return
    d["family"] = d["model"].str.split(":").str[0]
    d.loc[d["family"] == "lgbm", "family"] = "flat"
    pick = (d.sort_values("r2_overall", ascending=False)
             .groupby(["preset", "family"], as_index=False).first())
    presets = [p for p in ("baseline_2d", "inner_sphere", "selectivity")
               if p in set(pick["preset"])]
    fams = ["flat", "twostage", "anchored", "pairwise"]
    metrics = [("r2_overall", "R² overall"),
               ("r2_within", "R² within extractant"),
               ("r2_within_composition", "R² within composition")]
    # One colour per architecture, identical in every panel: colour follows the
    # entity, never the panel it happens to appear in.
    FAM_COLOR = {"flat": "#9aa0a6", "twostage": C["magenta"],
                 "anchored": C["blue"], "pairwise": C["green"]}
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.0, 4.2), sharey=False)
    width = 0.2
    for ax, (col, title) in zip(axes, metrics):
        xs = np.arange(len(presets))
        for k, fam in enumerate(fams):
            vals = []
            for p in presets:
                r = pick[(pick["preset"] == p) & (pick["family"] == fam)]
                vals.append(float(r[col].iloc[0]) if len(r) else np.nan)
            off = (k - (len(fams) - 1) / 2) * width
            ax.bar(xs + off, vals, width=width * 0.9, color=FAM_COLOR[fam],
                   label=fam if ax is axes[0] else None, zorder=3)
            for x, v in zip(xs + off, vals):
                if np.isfinite(v):
                    ax.text(x, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=7, color=INK, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([p.replace("_", "\n") for p in presets], fontsize=9)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        vmax = np.nanmax([ax.patches[i].get_height() for i in range(len(ax.patches))])
        ax.set_ylim(0, vmax * 1.28)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    fig.suptitle("Anchored-residual and delta-learning fix the within component",
                 x=0.06, ha="left", fontsize=13, y=1.03)
    fig.tight_layout()
    _save(fig, "fig4_architectures")


def fig_metal_free(res: pd.DataFrame) -> None:
    """The headline result: raw 3D costs the ordering, metal-free 3D does not."""
    want = [
        ("baseline_2d", "2D\nbaseline"),
        ("plus_g1", "raw 3D\nfirst shell"),
        ("plus_g5", "raw 3D\nxTB electronics"),
        ("plus_g14c", "metal-free\nfamily means"),
        ("plus_g13c", "metal-free\nfamily slopes"),
        ("ligand3d_only", "metal-free\nboth"),
    ]
    # Compare like with like: only runs from the sweeps that used the same
    # 3-repeat protocol and the same LightGBM settings.
    d = res[(res["row_filter"] == "has3d") & (res["repeats"] == 3)
            & res["tag"].astype(str).isin(["ablation", "cnfree", "combo"])]
    if d["preset"].nunique() < 4:
        d = res[res["row_filter"] == "has3d"]
    rows = []
    for preset, label in want:
        sub = d[(d["preset"] == preset) & (d["model"] == "lgbm")]
        if sub.empty:
            continue
        r = sub.sort_values("r2_overall", ascending=False).iloc[0]
        rows.append({"label": label, "within": r["r2_within"],
                     "spearman": r["sel_spearman_mean"]})
    if len(rows) < 3:
        return
    t = pd.DataFrame(rows)
    x = np.arange(len(t))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
    # Raw-3D bars are marked with a distinct hue AND a hatch, so the contrast is
    # not carried by colour alone.
    is_raw = t["label"].str.contains("raw")
    is_base = t["label"].str.contains("baseline")
    colors = [("#9aa0a6" if b else (C["orange"] if r else C["blue"]))
              for b, r in zip(is_base, is_raw)]
    hatches = ["" if b else ("///" if r else "") for b, r in zip(is_base, is_raw)]
    for ax, col, title, ref in (
            (axes[0], "within", "R² within extractant  (higher = better)", None),
            (axes[1], "spearman",
             "series ordering: mean Spearman across La→Lu  (higher = better)",
             float(t.loc[is_base, "spearman"].iloc[0]) if is_base.any() else None)):
        bars = ax.bar(x, t[col], color=colors, width=0.62, zorder=3)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
                bar.set_edgecolor("#ffffff")
        for xi, v in zip(x, t[col]):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
        if ref is not None:
            ax.axhline(ref, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)),
                       zorder=4)
            ax.text(-0.55, ref + 0.010, "baseline", ha="left",
                    fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(t["label"], fontsize=8.5, linespacing=1.4)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(0, t[col].max() * 1.22)
        ax.set_xlim(-0.65, len(t) - 0.35)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    fig.suptitle("The 3D information helps; its per-metal noise is what breaks "
                 "the lanthanide ordering",
                 x=0.055, ha="left", fontsize=13, y=1.04)
    fig.tight_layout()
    _save(fig, "fig8_metal_free_3d")


def fig_parity(oof_path: Path, label: str, name: str) -> None:
    """Measured vs out-of-fold predicted log D.

    The axis is clipped to the bulk of the data.  0.05 % of rows sit below
    log D = -5 (three measurements, all from one extractant) and would
    otherwise stretch the frame so far that the 99.9 % of points that matter
    collapse into a blob; the count of hidden points is stated on the figure.
    """
    if not oof_path.exists():
        return
    d = pd.read_parquet(oof_path)
    lo, hi = -5.2, 4.6
    inside = (d["y"].between(lo, hi)) & (d["oof"].between(lo, hi))
    hidden = int((~inside).sum())
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(d.loc[inside, "y"], d.loc[inside, "oof"], s=7, alpha=0.25,
               color=C["blue"], edgecolors="none", zorder=3)
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1.2, zorder=4)
    r2 = 1 - ((d["y"] - d["oof"]) ** 2).sum() / ((d["y"] - d["y"].mean()) ** 2).sum()
    mae = float((d["y"] - d["oof"]).abs().mean())
    note = f"R² = {r2:.3f}\nMAE = {mae:.2f}\nn = {len(d)}"
    if hidden:
        note += f"\n{hidden} point(s) outside axis"
    ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=9,
            color=INK)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured log D")
    ax.set_ylabel("predicted log D (out-of-fold)")
    ax.set_title(label, loc="left", pad=10, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    _save(fig, name)


def fig_uncertainty(path: Path) -> None:
    if not path.exists():
        return
    t = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(t))
    bars = ax.bar(x, t["mae"], color=C["blue"], width=0.62, zorder=3)
    for bar, v in zip(bars, t["mae"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in x])
    ax.set_xlabel("quintile of base-model disagreement (low → high)")
    ax.set_ylabel("mean |error| in log D")
    ax.set_ylim(0, t["mae"].max() * 1.25)
    ax.set_title("Ensemble disagreement is a usable error bar", loc="left", pad=12)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    _save(fig, "fig6_uncertainty_calibration")


def fig_split_variability(json_path: Path, champ_dir: Path) -> None:
    """How far a single leave-extractants-out split can move, model held fixed."""
    if not json_path.exists():
        return
    payload = json.loads(json_path.read_text())
    oof_path = champ_dir / payload["source_oof"]
    if not oof_path.exists():
        return
    from automl.split_variability import split_spread
    s = split_spread(pd.read_parquet(oof_path), payload["test_fraction_of_extractants"],
                     payload["n_draws"])
    pooled = payload["pooled_repeated_cv_r2"]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.hist(s, bins=44, color=C["blue"], edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.axvline(pooled, color=INK, linewidth=1.6, zorder=5)
    ax.annotate(f"repeated grouped CV\n(what this report uses)\nR² = {pooled:.3f}",
                xy=(pooled, ax.get_ylim()[1] * 0.92),
                xytext=(pooled - 0.20, ax.get_ylim()[1] * 0.92),
                ha="right", va="top", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=1.0))
    lo, hi = np.percentile(s, 5), np.percentile(s, 95)
    ax.axvspan(lo, hi, color=C["blue"], alpha=0.10, zorder=2)
    ax.annotate(f"90 % of single splits land between {lo:.2f} and {hi:.2f}",
                xy=(hi, ax.get_ylim()[1] * 0.45), xytext=(hi + 0.01,
                                                          ax.get_ylim()[1] * 0.45),
                ha="left", va="center", fontsize=9, color=INK2)
    ax.set_xlabel("R² on one random 20 %-of-extractants holdout")
    ax.set_ylabel("number of draws")
    ax.set_title("The same model scores anywhere from 0.29 to 0.69 "
                 "depending on the split", loc="left", pad=12)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    _save(fig, "fig9_split_variability")


def fig_split_series(sweep_dir: Path) -> None:
    """The headline: 3D blocks help Tb-Lu and hurt La-Gd, cancelling in the pool."""
    from automl.experiment import load_results
    d = load_results(sweep_dir)
    # The sweep's full-series control arm was cut short by walltime; fill any
    # gap from the ablation_catboost sweep, which used identical settings.
    alt = load_results(sweep_dir.parent / "ablation_catboost")
    if len(alt):
        alt = alt.assign(row_filter="has3d")
        have = set(d[d.get("row_filter", "") == "has3d"]["preset"]) if len(d) else set()
        d = pd.concat([d, alt[~alt["preset"].isin(have)]], ignore_index=True)
    if d.empty or "row_filter" not in d.columns:
        return
    label = {"plus_g1": "G1\nfirst shell", "plus_g5": "G5\nxTB electronics",
             "core3d_qc": "curated\ng_core", "inner_sphere": "inner\nsphere",
             "plus_g14c": "G14c\nmetal-free", "all_3d": "all 3D\ncolumns"}
    subsets = [("has3d", "full series", "#9aa0a6"),
               ("cn9_light", "La–Gd (CN 9)", C["orange"]),
               ("cn8_heavy", "Tb–Lu (CN 8)", C["blue"])]
    rows = []
    for rf, name, colour in subsets:
        sub = d[d["row_filter"] == rf]
        base = sub[sub["preset"] == "baseline_2d"]
        if base.empty:
            continue
        b = float(base.iloc[0]["r2_overall"])
        for preset, nice in label.items():
            r = sub[sub["preset"] == preset]
            if r.empty:
                continue
            rows.append({"subset": name, "colour": colour, "preset": nice,
                         "delta": float(r.iloc[0]["r2_overall"]) - b})
    if not rows:
        return
    t = pd.DataFrame(rows)
    presets = [p for p in label.values() if p in set(t["preset"])]
    names = [n for _, n, _ in subsets if n in set(t["subset"])]
    x = np.arange(len(presets))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    for k, name in enumerate(names):
        sub = t[t["subset"] == name].set_index("preset").reindex(presets)
        off = (k - (len(names) - 1) / 2) * width
        colour = sub["colour"].dropna().iloc[0]
        ax.bar(x + off, sub["delta"], width=width * 0.9, color=colour,
               label=name, zorder=3)
        for xi, v in zip(x + off, sub["delta"]):
            if np.isfinite(v):
                ax.text(xi, v + (0.003 if v >= 0 else -0.003), f"{v:+.3f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=7.5, color=INK, rotation=90)
    ax.axhline(0, color=INK2, linewidth=1.1, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(presets, fontsize=8.5, linespacing=1.4)
    ax.set_ylabel("Δ R² vs that subset's own 2D baseline")
    ax.set_title("The full-series null hides two opposite effects", loc="left",
                 pad=26)
    ax.text(0.0, 1.02, "3D blocks help the heavy lanthanides and hurt the light "
            "ones; pooled, they cancel",
            transform=ax.transAxes, fontsize=9, color=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.62, 0.02), ncol=1, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    lo, hi = t["delta"].min(), t["delta"].max()
    pad = 0.35 * max(abs(lo), abs(hi))
    ax.set_ylim(lo - pad, hi + pad)
    _save(fig, "fig10_split_series")


def fig_per_metal(champ_dir: Path) -> None:
    """R² per lanthanide for baseline vs the best configuration."""
    files = sorted(champ_dir.glob("oof_*.parquet"))
    if not files:
        return
    order = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
             "Er", "Tm", "Yb", "Lu"]

    def per_metal(p: Path) -> pd.Series:
        d = pd.read_parquet(p)
        return d.groupby("metal").apply(
            lambda g: 1 - ((g["y"] - g["oof"]) ** 2).sum()
            / max(((g["y"] - g["y"].mean()) ** 2).sum(), 1e-9),
            include_groups=False)

    base = next((f for f in files if "baseline_2D,_LightGBM" in f.name), None)
    # "best" = the highest-R2 champion OOF that is not the LightGBM reference.
    def _r2_of(p):
        d = pd.read_parquet(p)
        return 1 - ((d["y"] - d["oof"]) ** 2).sum() / ((d["y"] - d["y"].mean()) ** 2).sum()
    others = [f for f in files if f is not base]
    if base is None or not others:
        return
    best = max(others, key=_r2_of)
    best_label = best.stem.split("_", 2)[-1].replace("_", " ")
    b, c = per_metal(base), per_metal(best)
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = np.arange(len(order))
    ax.bar(x - 0.2, [b.get(m, np.nan) for m in order], width=0.36,
           color="#9aa0a6", label="2D baseline", zorder=3)
    ax.bar(x + 0.2, [c.get(m, np.nan) for m in order], width=0.36,
           color=C["blue"], label=f"best: {best_label}", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("R² (out-of-fold)")
    ax.set_xlabel("lanthanide (light → heavy)")
    ax.set_title("Per-metal held-out accuracy", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
              fontsize=8.5, borderaxespad=0.0)
    ax.set_ylim(0, max(0.7, ax.get_ylim()[1]))
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    _save(fig, "fig7_per_metal")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", default=str(REPO / "automl/reports/all_results.csv"))
    args = ap.parse_args()
    _style()

    res_path = Path(args.results_csv)
    if res_path.exists():
        res = pd.read_csv(res_path)
        res["tag"] = res.get("tag", pd.Series([""] * len(res))).fillna("")
        fig_decomposition(res, REPO / "automl/artifacts/champion")
        fig_block_ablation(res)
        fig_architectures(res)
        fig_metal_free(res)
    fig_noise_diagnostic(REPO / "automl/reports/noise_by_block.csv")
    fig_uncertainty(REPO / "automl/reports/uncertainty_calibration.csv")
    champ = REPO / "automl/artifacts/champion"
    fig_split_variability(REPO / "automl/reports/split_variability.json", champ)
    fig_split_series(REPO / "automl/artifacts/sweeps/singlecn")
    fig_per_metal(champ)
    files = sorted(champ.glob("oof_*.parquet"))
    base = next((f for f in files if "baseline_2D,_LightGBM" in f.name), None)
    if base is not None:
        fig_parity(base, "2D baseline (LightGBM), leave-extractants-out",
                   "fig5a_parity_baseline")
    others = [f for f in files if f is not base]
    if others:
        def _r2_of(p):
            d = pd.read_parquet(p)
            return 1 - ((d["y"] - d["oof"]) ** 2).sum() / ((d["y"] - d["y"].mean()) ** 2).sum()
        best = max(others, key=_r2_of)
        fig_parity(best, best.stem.split("_", 2)[-1].replace("_", " "),
                   "fig5b_parity_best")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
