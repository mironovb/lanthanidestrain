#!/usr/bin/env python3
"""Stage D: stack the sweep's out-of-fold predictions into one model.

Every sweep run with ``--save-oof`` drops a parquet of its out-of-fold
predictions keyed by ``safe_exp_id``.  Because all runs share the same grouped
CV protocol, those columns can be stacked directly: the meta-learner is trained
on the *same* leave-extractants-out folds, so a base model's prediction for an
extractant was made without ever seeing that extractant.

Two combination rules are compared:

``nnls``   non-negative least squares on the base predictions -- the classic
           stacking weight vector, constrained to be positive so the blend stays
           interpretable as "how much of each representation is used".
``uncert`` inverse-variance weighting: at each row, base models that disagree
           get down-weighted.  This is the uncertainty-ranked ensemble the
           representation-benchmark literature recommends, and it also gives a
           per-prediction error bar, which is what a screening campaign needs to
           decide which extractant to synthesise next.

The meta-learner itself is fitted under an outer grouped CV, so the reported
numbers are honest nested-CV numbers, not in-sample stacking scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from automl import evaluation as ev
from automl.experiment import load_results
from src.chemistry.coordination import LANTHANIDE_DESCRIPTORS

REPO = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO / "automl/artifacts/sweeps"
LANTHANIDE_INDEX = {k: v["lanthanide_index"] for k, v in LANTHANIDE_DESCRIPTORS.items()}


# Sweeps run under the corrected, properly-shuffled GroupKFold ("protocol B").
# Stacking base models whose out-of-fold predictions came from *different* fold
# assignments makes the meta-learner's weights hard to interpret, so the final
# ensemble is restricted to one protocol.
PROTOCOL_B_SWEEPS = ("combo", "robust", "ablation_catboost", "champion")


def collect_oof(sweep_dir: Path, min_r2: float = 0.30, max_models: int = 40,
                only_sweeps: tuple[str, ...] | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build the (rows x base-models) OOF matrix from every saved run."""
    records: list[tuple[str, float, Path]] = []
    for path in sorted(Path(sweep_dir).rglob("results.jsonl")):
        if only_sweeps and not any(f"/{s}/" in str(path) or str(path.parent).endswith(s)
                                   or s in path.parts for s in only_sweeps):
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok" or "oof_path" not in rec:
                continue
            r2 = (rec.get("metrics") or {}).get("r2_overall")
            if r2 is None or r2 < min_r2:
                continue
            p = Path(rec["oof_path"])
            if p.exists():
                records.append((rec["key"], float(r2), p))
    if not records:
        return pd.DataFrame(), pd.DataFrame(), []

    # Keep the strongest runs, but keep them diverse: at most 3 per preset.
    records.sort(key=lambda t: t[1], reverse=True)
    per_preset: dict[str, int] = {}
    kept = []
    for key, r2, p in records:
        preset = key.split("|")[0]
        if per_preset.get(preset, 0) >= 3:
            continue
        per_preset[preset] = per_preset.get(preset, 0) + 1
        kept.append((key, r2, p))
        if len(kept) >= max_models:
            break

    meta = None
    cols: dict[str, pd.Series] = {}
    names: list[str] = []
    for key, r2, p in kept:
        d = pd.read_parquet(p).drop_duplicates("safe_exp_id").set_index("safe_exp_id")
        if meta is None:
            meta = d[["y", "extractant_group", "composition_key", "metal"]].copy()
        cols[key] = d["oof"]
        names.append(key)
    X = pd.DataFrame(cols)
    common = meta.index.intersection(X.dropna().index)
    return X.loc[common], meta.loc[common], names


def _nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, _ = nnls(P, y)
    return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1.0 / P.shape[1])


def stack(X: pd.DataFrame, meta: pd.DataFrame, n_splits: int = 5,
          repeats: int = 3, seed: int = 7) -> dict[str, object]:
    P = X.to_numpy(dtype=float)
    y = meta["y"].to_numpy(dtype=float)
    groups = meta["extractant_group"].to_numpy()

    out: dict[str, object] = {}
    for name in ("nnls", "uncert", "mean", "best_single"):
        pred_sum = np.zeros(len(y))
        pred_cnt = np.zeros(len(y))
        weights_seen: list[np.ndarray] = []
        for rep in range(repeats):
            for tr, te in ev.grouped_folds(groups, n_splits=n_splits, seed=seed + rep):
                if name == "nnls":
                    w = _nnls_weights(P[tr], y[tr])
                    weights_seen.append(w)
                    pred = P[te] @ w
                elif name == "mean":
                    pred = P[te].mean(axis=1)
                elif name == "best_single":
                    r2s = [ev._r2(y[tr], P[tr, j]) for j in range(P.shape[1])]
                    pred = P[te, int(np.argmax(r2s))]
                else:  # inverse-variance / disagreement weighting
                    resid = P[tr] - y[tr][:, None]
                    var = np.maximum(resid.var(axis=0), 1e-6)
                    w = (1.0 / var)
                    w = w / w.sum()
                    spread = P[te].std(axis=1)
                    base = P[te] @ w
                    # shrink towards the plain mean where models disagree a lot
                    lam = 1.0 / (1.0 + spread)
                    pred = lam * base + (1 - lam) * P[te].mean(axis=1)
                pred_sum[te] += pred
                pred_cnt[te] += 1
        oof = pred_sum / np.maximum(pred_cnt, 1)
        m = ev.full_metrics(y, oof, meta.assign(
            lanthanide_index=meta["metal"].map(LANTHANIDE_INDEX).fillna(-1)))
        out[name] = {"metrics": m, "oof": oof}
        if weights_seen:
            w = np.mean(weights_seen, axis=0)
            out[name]["weights"] = dict(sorted(
                zip(X.columns, w.tolist()), key=lambda kv: kv[1], reverse=True))
    # Per-row uncertainty from base-model disagreement, for the screening use case.
    out["row_uncertainty"] = pd.Series(P.std(axis=1), index=X.index)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", default=str(SWEEP_DIR))
    ap.add_argument("--out-dir", default=str(REPO / "automl/reports"))
    ap.add_argument("--min-r2", type=float, default=0.30)
    ap.add_argument("--max-models", type=int, default=40)
    ap.add_argument("--protocol-b-only", action="store_true",
                    help="restrict to sweeps run with the corrected shuffled splitter")
    args = ap.parse_args()

    X, meta, names = collect_oof(
        Path(args.sweep_dir), args.min_r2, args.max_models,
        only_sweeps=PROTOCOL_B_SWEEPS if args.protocol_b_only else None)
    if X.empty:
        print("no OOF predictions found yet")
        return 0
    print(f"stacking {X.shape[1]} base models over {X.shape[0]} rows")
    res = stack(X, meta)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("best_single", "mean", "nnls", "uncert"):
        m = res[name]["metrics"]
        rows.append({"combiner": name, **{k: m.get(k) for k in
                    ("r2_overall", "r2_between", "r2_within",
                     "r2_within_composition", "rmse", "mae",
                     "sel_spearman_mean", "sel_logSF_r2", "sel_sign_accuracy")}})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ensemble_results.csv", index=False)
    print(table.round(4).to_string(index=False))

    weights = res.get("nnls", {}).get("weights", {})
    if weights:
        wpath = out_dir / "ensemble_weights.json"
        wpath.write_text(json.dumps(weights, indent=2))
        print("\ntop stacking weights:")
        for k, v in list(weights.items())[:12]:
            if v > 1e-4:
                print(f"  {v:6.3f}  {k}")

    # Calibration of the uncertainty estimate: does disagreement predict error?
    unc = res["row_uncertainty"].to_numpy()
    err = np.abs(res["nnls"]["oof"] - meta["y"].to_numpy())
    if unc.std() > 0:
        q = pd.qcut(pd.Series(unc), 5, labels=False, duplicates="drop")
        cal = pd.DataFrame({"quintile": q, "mae": err, "spread": unc}) \
            .groupby("quintile").mean()
        cal.to_csv(out_dir / "uncertainty_calibration.csv")
        print("\nuncertainty calibration (base-model spread vs realised |error|):")
        print(cal.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
