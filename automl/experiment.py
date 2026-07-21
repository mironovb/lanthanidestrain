#!/usr/bin/env python3
"""Cross-validated experiment runner for the leave-extractants-out protocol.

One "experiment" = (row subset, feature block preset, model, hyperparameters,
sample-weight scheme).  It is evaluated by repeated grouped K-fold on
``extractant_group`` and reported with the full metric decomposition from
``automl.evaluation``.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl import models as mz
from automl.dataset import BLOCK_PRESETS, GROUP_COL, TARGET, Blocks


@dataclass
class ExperimentSpec:
    preset: str = "baseline_2d"
    model: str = "lgbm"
    params: dict[str, Any] = field(default_factory=dict)
    weight_scheme: str = "none"
    n_splits: int = 5
    repeats: int = 2
    seed: int = 42
    row_filter: str = "all"          # all | ok_only | has3d
    target_clip: float = 0.0         # winsorise the *training* target at -clip
    tag: str = ""

    def key(self) -> str:
        return (f"{self.preset}|{self.model}|{self.weight_scheme}|{self.row_filter}"
                f"|k{self.n_splits}r{self.repeats}s{self.seed}"
                + (f"|clip{self.target_clip}" if self.target_clip else "")
                + (f"|{self.tag}" if self.tag else ""))


def apply_row_filter(df: pd.DataFrame, row_filter: str) -> pd.DataFrame:
    if row_filter in ("all", ""):
        return df
    if row_filter == "has3d":
        return df[df["has_3d"]].reset_index(drop=True)
    if row_filter == "ok_only":
        return df[df["geometry_ok"].astype(bool) & df["has_3d"]].reset_index(drop=True)
    # Single-CN subsets.  The stage-1 plan gives CN 9 to La-Gd and CN 8 to
    # Tb-Lu, so *within* either subset there is no coordination-number
    # staircase in the geometric descriptors.  Comparing "does 3D help" inside
    # one subset against the full series isolates the staircase artefact from
    # the single-conformer noise, without regenerating any geometry.
    if row_filter == "cn9_light":      # La..Gd, lanthanide_index 1-8
        return df[df["has_3d"] & (df["lanthanide_index"] <= 8)].reset_index(drop=True)
    if row_filter == "cn8_heavy":      # Tb..Lu, lanthanide_index 9-15
        return df[df["has_3d"] & (df["lanthanide_index"] >= 9)].reset_index(drop=True)
    if row_filter.startswith("cn9_matched"):
        # La-Gd, subsampled to match the Tb-Lu half in *ligand coverage*: same
        # number of extractants and a matched rows-per-extractant distribution.
        # This separates "3D helps the heavy half because it is chemistry" from
        # "3D helps the heavy half because its 2D model is data-starved".
        seed = int(row_filter.split(":")[1]) if ":" in row_filter else 0
        return _match_light_to_heavy(df, seed)
    raise ValueError(f"unknown row_filter {row_filter!r}")


def _match_light_to_heavy(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Subsample the La-Gd half to the Tb-Lu half's extractant-coverage profile.

    For every extractant in the heavy half, take the light-half extractant whose
    row count is closest and not yet used.  The result has the same number of
    extractants and a near-identical size distribution, so any remaining
    difference in what the 3D features buy is not explained by ligand coverage.
    """
    light = df[df["has_3d"] & (df["lanthanide_index"] <= 8)]
    heavy = df[df["has_3d"] & (df["lanthanide_index"] >= 9)]
    heavy_sizes = heavy.groupby(GROUP_COL).size().sort_values(ascending=False)
    light_sizes = light.groupby(GROUP_COL).size()
    rng = np.random.default_rng(seed)
    # break ties randomly so different seeds give different matched sets
    order = rng.permutation(len(light_sizes))
    pool = list(zip(light_sizes.index[order], light_sizes.to_numpy()[order]))
    chosen: list[str] = []
    for target in heavy_sizes.to_numpy():
        if not pool:
            break
        k = int(np.argmin([abs(sz - target) for _, sz in pool]))
        chosen.append(pool.pop(k)[0])
    return light[light[GROUP_COL].isin(chosen)].reset_index(drop=True)


def prepare_xy(df: pd.DataFrame, blocks: Blocks, preset: str
               ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    names = BLOCK_PRESETS.get(preset)
    if names is None:
        names = tuple(n.strip() for n in preset.split("+") if n.strip())
    cols = blocks.select(names)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{len(missing)} feature columns missing, e.g. {missing[:3]}")
    X = df[cols]
    y = df[TARGET].to_numpy(dtype=float)
    return X, y, cols


def _ligand_col_indices(cols: Sequence[str], blocks: Blocks) -> np.ndarray:
    """Column positions that describe the ligand only (for the two-stage level)."""
    ligand = set(blocks.mapping.get("rdkit", [])) | set(blocks.mapping.get("ecfp", []))
    idx = [i for i, c in enumerate(cols) if c in ligand]
    return np.asarray(idx if idx else list(range(len(cols))))


def _fit_predict_fold(spec: ExperimentSpec, Xv, y, tr, te, groups, comp,
                      weights, cols, blocks, seed, n_jobs, pair_keys=None):
    """One fold for either a plain model or one of the specialised architectures."""
    kind, _, base = spec.model.partition(":")
    base = base or "lgbm"
    w_tr = None if weights is None else weights[tr]
    pair_keys = pair_keys or {"binned": comp, "strict": comp}

    if kind == "twostage":
        from automl.advanced import TwoStageRegressor
        model = TwoStageRegressor(base_model=base, params=spec.params,
                                  level_model=spec.params.get("level_model", base),
                                  level_params=spec.params.get("level_params", {}),
                                  seed=seed, n_jobs=n_jobs)
        model.fit(Xv[tr], y[tr], groups=groups[tr],
                  ligand_cols=_ligand_col_indices(cols, blocks), sample_weight=w_tr)
        return np.asarray(model.predict(Xv[te], groups=groups[te]), dtype=float)

    if kind == "anchored":
        from automl.advanced import AnchoredResidualRegressor
        model = AnchoredResidualRegressor(
            base_model=base, params=spec.params,
            resid_model=spec.params.get("resid_model", base),
            resid_params=spec.params.get("resid_params", spec.params),
            level=spec.params.get("level", "extractant"),
            shape_weight=float(spec.params.get("shape_weight", 1.0)),
            seed=seed, n_jobs=n_jobs)
        model.fit(Xv[tr], y[tr], groups=groups[tr], composition=comp[tr],
                  sample_weight=w_tr)
        return np.asarray(model.predict(Xv[te], groups=groups[te],
                                        composition=comp[te]), dtype=float)

    if kind == "pairwise":
        from automl.advanced import PairwiseDeltaRegressor
        # Pairs are formed inside the *strict* composition block by default:
        # only the lanthanide differs, so every ligand/condition column cancels
        # exactly and the delta target is a true separation factor.
        pk = pair_keys[spec.params.get("pair_key", "strict")]
        model = PairwiseDeltaRegressor(
            base_model=base, params=spec.params,
            delta_model=spec.params.get("delta_model", base),
            delta_params=spec.params.get("delta_params", {}),
            delta_weight=float(spec.params.get("delta_weight", 1.0)),
            diff_only=bool(spec.params.get("diff_only", True)),
            seed=seed, n_jobs=n_jobs)
        model.fit(Xv[tr], y[tr], groups=groups[tr], composition=pk[tr],
                  sample_weight=w_tr)
        return np.asarray(model.predict(Xv[te], composition=pk[te]), dtype=float)

    model = mz.make_model(spec.model, spec.params, seed=seed, n_jobs=n_jobs)
    fit_kwargs = {}
    if w_tr is not None:
        fit_kwargs = ({f"{model.steps[-1][0]}__sample_weight": w_tr}
                      if hasattr(model, "steps") else {"sample_weight": w_tr})
    try:
        model.fit(Xv[tr], y[tr], **fit_kwargs)
    except TypeError:
        model.fit(Xv[tr], y[tr])
    return np.asarray(model.predict(Xv[te]), dtype=float)


def run_cv(df: pd.DataFrame, blocks: Blocks, spec: ExperimentSpec,
           n_jobs: int = 4, return_model: bool = False) -> ev.CVResult:
    """Repeated grouped K-fold; returns averaged out-of-fold predictions."""
    sub = apply_row_filter(df, spec.row_filter)
    X, y, cols = prepare_xy(sub, blocks, spec.preset)
    groups = sub[GROUP_COL].to_numpy()
    comp = sub["composition_key"].to_numpy()
    pair_keys = {"binned": comp,
                 "strict": sub.get("strict_composition_key", pd.Series(comp)).to_numpy(),
                 "extractant": groups}
    weights = mz.sample_weights(sub, spec.weight_scheme)

    base_family = spec.model.partition(":")[2] or spec.model
    needs_dense = base_family not in mz.NATIVE_NAN_MODELS
    Xv = X.to_numpy(dtype=np.float32) if not needs_dense else X.to_numpy(dtype=np.float64)

    # Winsorising the *training* target only.  0.8 % of rows sit below log D = -4
    # and carry 9 % of the total variance, 47 of them from a single extractant;
    # a held-out fold can never predict that tail, so letting it dominate the
    # squared-error loss costs accuracy everywhere else.  Scoring always uses the
    # true, unclipped y.
    y_fit = np.maximum(y, -abs(spec.target_clip)) if spec.target_clip else y

    oof_sum = np.zeros(len(sub))
    oof_cnt = np.zeros(len(sub))
    fold_r2: list[float] = []
    fold_metrics: list[dict[str, float]] = []
    t0 = time.time()
    for rep in range(spec.repeats):
        folds = ev.grouped_folds(groups, n_splits=spec.n_splits, seed=spec.seed + rep)
        for tr, te in folds:
            pred = _fit_predict_fold(spec, Xv, y_fit, tr, te, groups, comp,
                                     weights, cols, blocks, spec.seed + rep,
                                     n_jobs, pair_keys=pair_keys)
            oof_sum[te] += pred
            oof_cnt[te] += 1
            fold_r2.append(ev._r2(y[te], pred))
            fold_metrics.append(ev.variance_decomposed_r2(y[te], pred, groups[te]))

    oof = np.divide(oof_sum, np.maximum(oof_cnt, 1))
    metrics = ev.full_metrics(y, oof, sub)
    metrics["n_rows"] = float(len(sub))
    metrics["n_features"] = float(len(cols))
    metrics["n_groups"] = float(len(np.unique(groups)))
    metrics["fit_seconds"] = float(time.time() - t0)
    extra: dict[str, Any] = {"columns": cols}
    if return_model and ":" not in spec.model:
        final = mz.make_model(spec.model, spec.params, seed=spec.seed, n_jobs=n_jobs)
        fk = {}
        if weights is not None:
            fk = ({f"{final.steps[-1][0]}__sample_weight": weights}
                  if hasattr(final, "steps") else {"sample_weight": weights})
        try:
            final.fit(Xv, y, **fk)
        except TypeError:
            final.fit(Xv, y)
        extra["model"] = final
        extra["X"] = X
    return ev.CVResult(oof=oof, metrics=metrics, fold_metrics=fold_metrics,
                       per_fold_r2=fold_r2, extra=extra)


def run_and_record(df: pd.DataFrame, blocks: Blocks, spec: ExperimentSpec,
                   out_dir: Path, n_jobs: int = 4, save_oof: bool = False
                   ) -> dict[str, Any]:
    """Run one experiment and append a JSON line to ``out_dir/results.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"spec": asdict(spec), "key": spec.key()}
    try:
        res = run_cv(df, blocks, spec, n_jobs=n_jobs)
        record["metrics"] = {k: (None if not np.isfinite(v) else float(v))
                             for k, v in res.summary_row().items()}
        record["status"] = "ok"
        if save_oof:
            sub = apply_row_filter(df, spec.row_filter)
            digest = hashlib.sha1(spec.key().encode()).hexdigest()[:16]
            oof_path = out_dir / f"oof_{digest}.parquet"
            pd.DataFrame({
                "safe_exp_id": sub["safe_exp_id"].to_numpy(),
                "y": sub[TARGET].to_numpy(),
                "oof": res.oof,
                "extractant_group": sub[GROUP_COL].to_numpy(),
                "composition_key": sub["composition_key"].to_numpy(),
                "metal": sub["metal"].to_numpy(),
            }).to_parquet(oof_path, index=False)
            record["oof_path"] = str(oof_path)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=4)
    with open(out_dir / "results.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def load_results(out_dir: Path) -> pd.DataFrame:
    """Flatten every results.jsonl under ``out_dir`` into a dataframe."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(out_dir).rglob("results.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") != "ok":
                    continue
                row = dict(rec["spec"])
                row["key"] = rec["key"]
                row["source"] = str(path)
                row.update(rec.get("metrics", {}))
                row.pop("params", None)
                row["params_json"] = json.dumps(rec["spec"].get("params", {}))
                rows.append(row)
    return pd.DataFrame(rows)
