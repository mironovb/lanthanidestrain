#!/usr/bin/env python3
"""Paired comparison of champion configurations.

The absolute bootstrap interval on R^2 is wide -- roughly [0.39, 0.64] for a
model scoring 0.53 -- because resampling 190 extractants, one of which is 29 %
of the table, moves the *target variance* as much as the error.  That interval
answers "how well would this model score on a different draw of extractants",
which is not the question being asked.

The question being asked is "does configuration B beat configuration A", and
for that the right statistic is the **paired** difference: resample extractants
once and score both models on the *same* resample, using out-of-fold predictions
that came from the same folds.  The shared variance cancels and the interval
collapses to something informative.

Outputs a matrix of paired deltas against the chosen reference configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev

REPO = Path(__file__).resolve().parents[1]
CHAMP_DIR = REPO / "automl/artifacts/champion"


def load_champion(champ_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    oof: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for j in sorted(champ_dir.glob("champ_*.json")):
        rec = json.loads(j.read_text())
        label = rec["label"]
        safe = label.replace(" ", "_").replace("/", "-")
        matches = list(champ_dir.glob(f"oof_*_{safe}.parquet"))
        if not matches:
            continue
        d = pd.read_parquet(matches[0]).drop_duplicates("safe_exp_id")
        oof[label] = d.set_index("safe_exp_id")
        meta[label] = rec
    return oof, meta


def _group_mean(v: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Per-row mean of ``v`` within its group, via bincount (no pandas)."""
    sums = np.bincount(codes, weights=v, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    counts[counts == 0] = 1.0
    return (sums / counts)[codes]


def _metrics_fast(y, p, gcodes, ng, ccodes, nc) -> dict[str, float]:
    """The four decomposed R^2 values plus MAE, computed with integer codes.

    The bootstrap calls this ~400 times per pair, so the pandas groupby version
    is far too slow; this is the same arithmetic on bincount.
    """
    out: dict[str, float] = {}
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out["r2_overall"] = 1 - float(np.sum((y - p) ** 2)) / ss_tot if ss_tot > 0 else np.nan

    gy = _group_mean(y, gcodes, ng)
    gp = _group_mean(p, gcodes, ng)
    # between: size-weighted R^2 of the group means == R^2 of the per-row means
    ssb = float(np.sum((gy - y.mean()) ** 2))
    out["r2_between"] = 1 - float(np.sum((gy - gp) ** 2)) / ssb if ssb > 0 else np.nan
    yc, pc = y - gy, p - gp
    ssw = float(np.sum(yc ** 2))
    out["r2_within"] = 1 - float(np.sum((yc - pc) ** 2)) / ssw if ssw > 0 else np.nan

    cy = _group_mean(y, ccodes, nc)
    cp = _group_mean(p, ccodes, nc)
    ycc, pcc = y - cy, p - cp
    ssc = float(np.sum(ycc ** 2))
    out["r2_within_composition"] = (1 - float(np.sum((ycc - pcc) ** 2)) / ssc
                                    if ssc > 0 else np.nan)
    out["mae"] = float(np.mean(np.abs(y - p)))
    return out


def _metrics(y, p, group, comp) -> dict[str, float]:
    gcodes, guniq = pd.factorize(group)
    ccodes, cuniq = pd.factorize(comp)
    return _metrics_fast(y, p, gcodes, len(guniq), ccodes, len(cuniq))


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 400,
                     seed: int = 0) -> dict[str, tuple[float, float, float, float]]:
    """Return {metric: (delta, lo, hi, P(delta>0))} for b minus a."""
    common = a.index.intersection(b.index)
    # Two configurations scored on disjoint row subsets (e.g. the La-Gd and
    # Tb-Lu single-CN runs) cannot be paired; say so instead of returning a
    # number computed on a handful of shared rows.
    if len(common) < 0.5 * min(len(a), len(b)):
        return {}
    a, b = a.loc[common], b.loc[common]
    y = a["y"].to_numpy(dtype=float)
    pa = a["oof"].to_numpy(dtype=float)
    pb = b["oof"].to_numpy(dtype=float)
    groups = a["extractant_group"].to_numpy()
    comp = a["composition_key"].to_numpy()

    keys = ("r2_overall", "r2_between", "r2_within", "r2_within_composition", "mae")
    gcodes, guniq = pd.factorize(groups)
    ccodes, cuniq = pd.factorize(comp)
    ng, nc = len(guniq), len(cuniq)
    point = {k: _metrics_fast(y, pb, gcodes, ng, ccodes, nc).get(k, np.nan)
             - _metrics_fast(y, pa, gcodes, ng, ccodes, nc).get(k, np.nan)
             for k in keys}

    rng = np.random.default_rng(seed)
    index = [np.flatnonzero(gcodes == g) for g in range(ng)]
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, ng, size=ng)
        idx = np.concatenate([index[g] for g in pick])
        # Re-code groups per draw: a group sampled twice must count twice, so
        # the codes are rebuilt from the draw rather than reused.
        gc = np.repeat(np.arange(ng), [len(index[g]) for g in pick])
        cc = pd.factorize(comp[idx])[0]
        ma = _metrics_fast(y[idx], pa[idx], gc, ng, cc, cc.max() + 1)
        mb = _metrics_fast(y[idx], pb[idx], gc, ng, cc, cc.max() + 1)
        for k in keys:
            va, vb = ma.get(k, np.nan), mb.get(k, np.nan)
            if np.isfinite(va) and np.isfinite(vb):
                draws[k].append(vb - va)
    out = {}
    for k in keys:
        d = np.asarray(draws[k])
        if d.size == 0:
            continue
        out[k] = (point[k], float(np.percentile(d, 5)), float(np.percentile(d, 95)),
                  float(np.mean(d > 0)))
    return out


def load_sweep(sweep_dir: Path, min_r2: float = -1.0
               ) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Load OOF predictions from an ordinary sweep (needs --save-oof)."""
    oof: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for path in sorted(Path(sweep_dir).rglob("results.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok" or "oof_path" not in rec:
                continue
            p = Path(rec["oof_path"])
            if not p.exists():
                continue
            m = rec.get("metrics") or {}
            if (m.get("r2_overall") or -9) < min_r2:
                continue
            # Label must carry the row filter: a sweep can score the same
            # preset on different row subsets (e.g. the single-CN test), and
            # collapsing those onto one label would silently drop rows.
            spec = rec["spec"]
            rf = spec.get("row_filter", "")
            label = spec["preset"] if rf in ("", "has3d") else f"{rf}/{spec['preset']}"
            if label in oof:               # keep the best run per label
                if m.get("r2_overall", -9) <= meta[label]["metrics"].get("r2_overall", -9):
                    continue
            oof[label] = pd.read_parquet(p).drop_duplicates(
                "safe_exp_id").set_index("safe_exp_id")
            meta[label] = {"label": label, "preset": spec["preset"],
                           "model": spec["model"], "metrics": m}
    return oof, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default="",
                    help="compare presets inside one sweep instead of the champion set")
    ap.add_argument("--champ-dir", default=str(CHAMP_DIR))
    ap.add_argument("--reference", default="baseline 2D, LightGBM")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", default=str(REPO / "automl/reports/paired_comparison.csv"))
    args = ap.parse_args()

    if args.sweep_dir:
        oof, meta = load_sweep(Path(args.sweep_dir))
    else:
        oof, meta = load_champion(Path(args.champ_dir))
    if args.reference not in oof:
        if not oof:
            print("no champion results yet")
            return 0
        args.reference = sorted(oof)[0]
        print(f"reference not found; using {args.reference!r}")
    ref = oof[args.reference]

    rows = []
    for label, d in oof.items():
        m = meta[label]["metrics"]
        row = {"config": label,
               "preset": meta[label]["preset"], "model": meta[label]["model"],
               "r2_overall": m.get("r2_overall"),
               "r2_between": m.get("r2_between"),
               "r2_within": m.get("r2_within"),
               "r2_within_composition": m.get("r2_within_composition"),
               "sel_logSF_r2": m.get("sel_logSF_r2"),
               "sel_spearman_mean": m.get("sel_spearman_mean"),
               "mae": m.get("mae")}
        if label != args.reference:
            pb = paired_bootstrap(ref, d, n_boot=args.n_boot)
            if not pb:
                row["not_comparable"] = True
            for k, (delta, lo, hi, p) in pb.items():
                row[f"d_{k}"] = delta
                row[f"d_{k}_lo"] = lo
                row[f"d_{k}_hi"] = hi
                row[f"d_{k}_p_better"] = p
        rows.append(row)

    t = pd.DataFrame(rows).sort_values("r2_overall", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)

    show = ["config", "r2_overall", "r2_between", "r2_within",
            "r2_within_composition", "sel_logSF_r2", "mae"]
    print(f"=== champion configurations (reference = {args.reference!r}) ===")
    print(t[show].round(4).to_string(index=False))
    print()
    print("=== paired delta vs reference (90 % interval, P(better)) ===")
    d = t[t["config"] != args.reference]
    if "d_r2_overall" in d.columns:
        view = d[["config", "d_r2_overall", "d_r2_overall_lo", "d_r2_overall_hi",
                  "d_r2_overall_p_better", "d_r2_within", "d_r2_within_p_better"]]
        print(view.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
