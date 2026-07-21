#!/usr/bin/env python3
"""Why does the published FCNN score +0.005 on adjacent pairs?

**Post-hoc diagnostic, not part of the pre-registered factorial.**  It does not
touch the primary endpoint; it explains one term in the attribution.

The factorial's largest term is the gap between the published FCNN baseline and
the same feature set trained in the topological harness.  Calling that "tuning"
would be vague and unhelpful, so this isolates the specific cause.

First hypothesis, from reading ``automl/models.py:244`` -- and **it was wrong**,
which is recorded here rather than quietly dropped.  The baseline uses
``early_stopping=True, validation_fraction=0.12``, and sklearn holds out a
random 12 % of *rows*, not of *extractants*.  Under a leave-extractants-out
protocol that looked like a mis-specified stopping criterion: the stopping
signal measures within-extractant interpolation while the model is scored on
unseen extractants.  (Nothing leaks into the test fold -- the outer split is
still grouped -- so the published number was always honest; the question was
only whether the model stops in the right place.)

Replacing it with a group-held-out stopping split scored **-0.0045**, slightly
*worse* than the published +0.005.  So the stopping split is not the cause.

Second hypothesis, still open at the time of writing: the ``QuantileTransformer``
in the shared dense pipeline (``models.py:127``).  It maps each feature to its
rank and then to a Gaussian -- monotone, so ordering survives, but *spacing*
does not.  Adjacent-lanthanide selectivity is entirely a question about spacing:
neighbouring ionic radii differ by ~0.013 A, and a rank transform spreads the 14
distinct radii in this dataset to roughly equal intervals no matter how close
together they really are.  A gradient-boosted tree is invariant to that (it only
ever compares); a network asked to predict a *difference* is not.  That would
also explain why CatBoost, sharing the same feature block, does not suffer.

Variants, identical features, rows and outer folds:

    published    as reported: quantile transform, early_stopping=True
    grouped      stopped on a GROUP-held-out split          -> tested, not it
    no_stop      trained to max_iter, no early stopping
    std_scaler   the published pipeline with StandardScaler instead
    ensemble16   the published configuration, 16 seeds averaged
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from automl import evaluation as ev
from automl.dataset import BLOCK_PRESETS, GROUP_COL, TARGET
from automl.matrix_cache import load_cache
from automl.topo.train import build_row_table

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/fcnn_diagnostic.json"


def _pipe(seed: int, early: bool, max_iter: int = 400,
          scaler: str = "quantile") -> Pipeline:
    """The published pipeline, exactly (models.py:127 and :244).

    ``scaler`` is the one deliberate deviation.  The published pipeline applies
    a QuantileTransformer, which maps each feature to its rank and then to a
    Gaussian -- monotone, so ordering survives, but *spacing* does not.
    Adjacent-lanthanide selectivity is a question about spacing: neighbouring
    ionic radii differ by ~0.013 A, and a rank transform maps the 14 distinct
    radii in this dataset to 14 roughly equally spaced values regardless of how
    close together they really are.  A tree does not care; a network asked to
    predict a *difference* does.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", QuantileTransformer(output_distribution="normal",
                                      n_quantiles=500, subsample=100000,
                                      random_state=0)
         if scaler == "quantile" else StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(256, 128), alpha=1e-3,
            learning_rate_init=1e-3, batch_size=128, max_iter=max_iter,
            early_stopping=early, n_iter_no_change=25, validation_fraction=0.12,
            random_state=seed)),
    ])


def _fit_predict(X, y, tr, te, *, seed, mode, groups):
    if mode == "grouped":
        # Same budget, same architecture -- only the *composition* of the
        # stopping split changes: whole extractants instead of random rows.
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups[tr])
        val_g = set(rng.choice(uniq, max(1, int(0.12 * len(uniq))),
                               replace=False).tolist())
        is_val = np.array([g in val_g for g in groups[tr]])
        fit, val = tr[~is_val], tr[is_val]
        best, best_it, patience = np.inf, 0, 0
        pipe = _pipe(seed, early=False, max_iter=1)
        pipe.named_steps["model"].set_params(warm_start=True, max_iter=1)
        Xf = pipe[:-1].fit_transform(X[fit])
        Xv, Xt = pipe[:-1].transform(X[val]), pipe[:-1].transform(X[te])
        m = pipe.named_steps["model"]
        best_pred = None
        for _ in range(400):
            m.fit(Xf, y[fit])
            v = float(np.mean((m.predict(Xv) - y[val]) ** 2))
            if v < best - 1e-6:
                best, patience, best_pred = v, 0, m.predict(Xt)
            else:
                patience += 1
                if patience >= 25:
                    break
        return best_pred if best_pred is not None else m.predict(Xt)
    pipe = _pipe(seed, early=(mode in ("published", "std_scaler")),
                 max_iter=400 if mode != "no_stop" else 1200,
                 scaler="standard" if mode == "std_scaler" else "quantile")
    pipe.fit(X[tr], y[tr])
    return pipe.predict(X[te])


def run(mode: str, seeds: list[int], folds: int, repeats: int) -> dict:
    df, X, _cols = build_row_table("baseline_2d", "snn")
    y = df[TARGET].to_numpy(dtype=float)
    groups = df[GROUP_COL].to_numpy()
    comp = df["composition_key"].to_numpy()
    li = df["lanthanide_index"].to_numpy()

    acc = np.zeros(len(df))
    for s in seeds:
        oof_sum, oof_cnt = np.zeros(len(df)), np.zeros(len(df))
        for rep in range(repeats):
            for tr, te in ev.grouped_folds(groups, n_splits=folds, seed=42 + rep):
                oof_sum[te] += _fit_predict(X, y, tr, te, seed=s, mode=mode,
                                            groups=groups)
                oof_cnt[te] += 1
        acc += oof_sum / np.maximum(oof_cnt, 1)
    oof = acc / len(seeds)
    m = ev.adjacent_pair_metrics(y, oof, comp, li)
    return {"mode": mode, "n_seeds": len(seeds),
            "adj_r2": float(m.get("sel_adj_logSF_r2", np.nan)),
            "r2_overall": float(ev._r2(y, oof)),
            "n_rows": int(len(df))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", nargs="*",
                    default=["published", "grouped", "no_stop", "std_scaler",
                             "ensemble16"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    # The published baseline used seed 42 and a single model.
    seeds16 = [7, 11, 23, 37, 42, 51, 67, 83,
               211, 223, 233, 241, 251, 263, 271, 281]
    out = []
    for mode in args.modes:
        t0 = time.time()
        r = run("published" if mode == "ensemble16" else mode,
                seeds16 if mode == "ensemble16" else [42],
                args.folds, args.repeats)
        r["mode"] = mode
        r["seconds"] = time.time() - t0
        out.append(r)
        print(f"  {mode:11s} adjR2 = {r['adj_r2']:+.4f}   "
              f"R2 = {r['r2_overall']:+.4f}   "
              f"[{r['seconds']/60:.1f} min]", flush=True)

    # One file per mode: the four modes run as separate array tasks, and a
    # single shared path would have them overwrite each other with whichever
    # finished last -- silently reporting one mode's number under four labels.
    dest = (OUT if len(args.modes) > 1
            else OUT.with_name(f"fcnn_diagnostic_{args.modes[0]}.json"))
    dest.write_text(json.dumps(out, indent=2))
    print(f"[fcnn-diag] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
