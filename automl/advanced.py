#!/usr/bin/env python3
"""Architectures aimed specifically at the *within-extractant* deficit.

The baseline's problem is not accuracy in bulk, it is that almost all of its
skill is "which extractant is this" (R^2_between ~ 0.71-0.81) and very little is
"how does this extractant discriminate La from Lu under these conditions"
(R^2_within ~ 0.22-0.26).  Feeding a flat model more columns does not fix that:
a single squared-error objective spends its capacity on the between component,
which carries most of the variance.

Two estimators here attack the within component directly.

``TwoStageRegressor``
    Stage 1 predicts the extractant-level mean log D from ligand-level features
    only (one training row per extractant, so TODGA cannot dominate).  Stage 2
    predicts the residual ``y - mean_extractant(y)`` from everything.  The
    prediction is the sum.  Because stage 2's target already has the between
    component removed, its loss is entirely about within-extractant structure.

``PairwiseDeltaRegressor``
    Delta learning.  Inside one composition block -- same extractant, same acid,
    same diluent, same temperature, only the lanthanide changes -- every ligand
    and condition feature is identical, so the *difference* between two rows is
    a pure metal-and-geometry object.  Training on ordered pairs
    (x_i, x_j) -> y_i - y_j therefore learns separation factors directly, and
    turns ~n rows per block into ~n(n-1) training examples.  At prediction time
    the block's absolute level comes from a base regressor and the shape comes
    from the averaged pairwise predictions:

        y_hat_i = base_block_mean + (1/n) * sum_j delta_hat(i, j)

    which is the least-squares reconstruction of a set of values from all of
    their pairwise differences, anchored on the base model's block mean.

Transduction caveat -- read this before quoting the numbers
-----------------------------------------------------------
``AnchoredResidualRegressor`` and ``PairwiseDeltaRegressor`` are *batch*
predictors: the prediction for one row depends on which other rows are being
predicted alongside it, because the anchor is the mean base prediction over the
extractant (or composition block) and the delta reconstruction averages over
that block's members.

No target value of any test row is used anywhere, so there is no label leakage
and the leave-extractants-out guarantee is intact.  But these are not
row-independent predictors, and they are only valid for the use case they were
built for: scoring a whole candidate extractant across the lanthanide series in
one batch, which is exactly what a screening campaign does.  For single-row
inference the flat model is the honest baseline, and the flat numbers are
reported alongside throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from automl import models as mz


def _fit(model, X, y, w=None):
    kwargs = {}
    if w is not None:
        kwargs = ({f"{model.steps[-1][0]}__sample_weight": w}
                  if hasattr(model, "steps") else {"sample_weight": w})
    try:
        model.fit(X, y, **kwargs)
    except TypeError:
        model.fit(X, y)
    return model


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """Extractant-level mean + within-extractant residual."""

    def __init__(self, base_model: str = "lgbm", params: dict | None = None,
                 level_model: str = "lgbm", level_params: dict | None = None,
                 seed: int = 0, n_jobs: int = 4):
        self.base_model = base_model
        self.params = params or {}
        self.level_model = level_model
        self.level_params = level_params or {}
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            ligand_cols: np.ndarray | None = None, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        frame = pd.DataFrame({"g": groups, "y": y})
        gmean = frame.groupby("g")["y"].mean()

        # --- stage 1: one row per extractant -------------------------------
        lig = ligand_cols if ligand_cols is not None else np.arange(X.shape[1])
        self.ligand_cols_ = np.asarray(lig)
        agg = pd.DataFrame(X[:, self.ligand_cols_]).groupby(groups).mean()
        self.level_ = mz.make_model(self.level_model, self.level_params,
                                    seed=self.seed, n_jobs=self.n_jobs)
        _fit(self.level_, agg.to_numpy(), gmean.loc[agg.index].to_numpy())

        # --- stage 2: residual after removing the true group mean ----------
        resid = y - frame["g"].map(gmean).to_numpy()
        self.resid_ = mz.make_model(self.base_model, self.params,
                                    seed=self.seed, n_jobs=self.n_jobs)
        _fit(self.resid_, X, resid, sample_weight)
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        agg = pd.DataFrame(X[:, self.ligand_cols_]).groupby(groups).mean()
        level = pd.Series(self.level_.predict(agg.to_numpy()), index=agg.index)
        base = pd.Series(groups).map(level).to_numpy()
        base = np.where(np.isfinite(base), base, self.global_mean_)
        return base + self.resid_.predict(X)


class PairwiseDeltaRegressor(BaseEstimator, RegressorMixin):
    """Delta learning on metal pairs inside a fixed composition block."""

    def __init__(self, base_model: str = "lgbm", params: dict | None = None,
                 delta_model: str = "lgbm", delta_params: dict | None = None,
                 seed: int = 0, n_jobs: int = 4, max_pairs: int = 400_000,
                 delta_weight: float = 1.0, diff_only: bool = True):
        self.base_model = base_model
        self.params = params or {}
        self.delta_model = delta_model
        self.delta_params = delta_params or {}
        self.seed = seed
        self.n_jobs = n_jobs
        self.max_pairs = max_pairs
        self.delta_weight = delta_weight
        self.diff_only = diff_only

    # -- pair construction --------------------------------------------------
    @staticmethod
    def _block_pairs(comp: np.ndarray, rng: np.random.Generator,
                     max_pairs: int) -> tuple[np.ndarray, np.ndarray]:
        ii, jj = [], []
        order = pd.Series(np.arange(len(comp))).groupby(comp).apply(list)
        for idx in order:
            idx = np.asarray(idx)
            if len(idx) < 2:
                continue
            a, b = np.meshgrid(idx, idx, indexing="ij")
            mask = a != b
            ii.append(a[mask])
            jj.append(b[mask])
        if not ii:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        ii = np.concatenate(ii)
        jj = np.concatenate(jj)
        if len(ii) > max_pairs:
            sel = rng.choice(len(ii), size=max_pairs, replace=False)
            ii, jj = ii[sel], jj[sel]
        return ii, jj

    def _pair_features(self, X: np.ndarray, ii: np.ndarray, jj: np.ndarray
                       ) -> np.ndarray:
        diff = X[ii] - X[jj]
        if self.diff_only:
            # Columns that never differ inside a block (ligand identity,
            # conditions) collapse to exactly zero and are dropped once at fit
            # time, leaving a compact metal/geometry-only design matrix.
            return diff[:, self.pair_cols_]
        return np.hstack([diff[:, self.pair_cols_], X[ii], X[jj]])

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            composition: np.ndarray, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.seed)

        # base model: absolute level, ordinary regression on all rows
        self.base_ = mz.make_model(self.base_model, self.params,
                                   seed=self.seed, n_jobs=self.n_jobs)
        _fit(self.base_, X, y, sample_weight)

        ii, jj = self._block_pairs(np.asarray(composition), rng, self.max_pairs)
        self.n_pairs_ = int(len(ii))
        if self.n_pairs_ < 50:
            self.delta_ = None
            return self
        raw = X[ii] - X[jj]
        with np.errstate(invalid="ignore"):
            varying = np.nanstd(raw, axis=0) > 1e-9
        self.pair_cols_ = np.flatnonzero(varying)
        if self.pair_cols_.size == 0:
            self.delta_ = None
            return self
        P = self._pair_features(X, ii, jj)
        dy = y[ii] - y[jj]
        self.delta_ = mz.make_model(self.delta_model, self.delta_params,
                                    seed=self.seed, n_jobs=self.n_jobs)
        _fit(self.delta_, P, dy,
             None if sample_weight is None else sample_weight[ii] * sample_weight[jj])
        return self

    def predict(self, X: np.ndarray, composition: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        base = np.asarray(self.base_.predict(X), dtype=float)
        if getattr(self, "delta_", None) is None:
            return base
        comp = pd.Series(np.asarray(composition))
        out = base.copy()
        for _, idx in comp.groupby(comp).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) < 2:
                continue
            a, b = np.meshgrid(idx, idx, indexing="ij")
            mask = a != b
            ii, jj = a[mask], b[mask]
            P = self._pair_features(X, ii, jj)
            d = np.asarray(self.delta_.predict(P), dtype=float)
            # least-squares reconstruction: value_i = mean + row-mean of deltas
            n = len(idx)
            dmat = np.zeros((n, n))
            pos = {int(v): k for k, v in enumerate(idx)}
            for k in range(len(ii)):
                dmat[pos[int(ii[k])], pos[int(jj[k])]] = d[k]
            shape = dmat.sum(axis=1) / n
            shape = shape - shape.mean()
            anchor = base[idx].mean()
            out[idx] = anchor + (1 - self.delta_weight) * (base[idx] - anchor) \
                + self.delta_weight * shape
        return out


class AnchoredResidualRegressor(BaseEstimator, RegressorMixin):
    """Level from the flat model, shape from a within-specialised model.

    The plain two-stage estimator loses R^2_between because its stage-1 level
    model only ever sees ~190 training rows (one per extractant).  The flat
    model is much better at that job.  So keep the flat model for the *level*
    and only replace the part it is bad at:

        y_hat_i = mean_over_block(flat_pred) + centred_residual_hat_i

    The residual model is trained on ``y - mean_extractant(y)``, i.e. a target
    from which the entire between-extractant component has been removed, so its
    squared-error loss is spent exclusively on within-extractant structure.
    ``level='composition'`` anchors on the extractant+conditions block instead,
    which isolates the lanthanide series even further.
    """

    def __init__(self, base_model: str = "lgbm", params: dict | None = None,
                 resid_model: str = "lgbm", resid_params: dict | None = None,
                 level: str = "extractant", shape_weight: float = 1.0,
                 seed: int = 0, n_jobs: int = 4):
        self.base_model = base_model
        self.params = params or {}
        self.resid_model = resid_model
        self.resid_params = resid_params or {}
        self.level = level
        self.shape_weight = shape_weight
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(self, X, y, groups, composition=None, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        key = groups if self.level == "extractant" else composition
        self.base_ = _fit(mz.make_model(self.base_model, self.params,
                                        seed=self.seed, n_jobs=self.n_jobs),
                          X, y, sample_weight)
        resid = y - pd.Series(y).groupby(pd.Series(key)).transform("mean").to_numpy()
        self.resid_ = _fit(mz.make_model(self.resid_model, self.resid_params,
                                         seed=self.seed, n_jobs=self.n_jobs),
                           X, resid, sample_weight)
        return self

    def predict(self, X, groups, composition=None):
        X = np.asarray(X, dtype=float)
        key = pd.Series(groups if self.level == "extractant" else composition)
        base = pd.Series(np.asarray(self.base_.predict(X), dtype=float))
        shape = pd.Series(np.asarray(self.resid_.predict(X), dtype=float))
        anchor = base.groupby(key).transform("mean").to_numpy()
        # Centre the residual model inside each block so it only supplies shape.
        shape_c = (shape - shape.groupby(key).transform("mean")).to_numpy()
        base_shape = (base - base.groupby(key).transform("mean")).to_numpy()
        w = self.shape_weight
        return anchor + w * shape_c + (1.0 - w) * base_shape


class BlendRegressor(BaseEstimator, RegressorMixin):
    """Convex blend of the flat model and a within-specialised model."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def combine(self, flat: np.ndarray, special: np.ndarray) -> np.ndarray:
        return (1 - self.alpha) * flat + self.alpha * special
