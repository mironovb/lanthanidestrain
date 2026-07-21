#!/usr/bin/env python3
"""Model zoo + preprocessing + sample-weighting schemes for the AutoML sweep.

Every model is wrapped so that the driver only ever calls ``fit`` and
``predict``; the wrapper owns whatever imputation/scaling that family needs.
Gradient-boosted trees consume NaN natively, which matters here because the
3D blocks are legitimately missing for ~1% of rows and the CShM columns are
missing whenever the coordination number has no matching reference polyhedron.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAS_LGBM = _has("lightgbm")
HAS_XGB = _has("xgboost")
HAS_CATBOOST = _has("catboost")


# ---------------------------------------------------------------------------
# Sample weighting
# ---------------------------------------------------------------------------
def sample_weights(df: pd.DataFrame, scheme: str) -> np.ndarray | None:
    """Training weights.

    ``group_inv``    counteracts the fact that one extractant is 29% of the
                     table, so the loss is not dominated by TODGA.
    ``metal_inv``    the column shipped with the dataset (Eu is 26%).
    ``target_lds``   label-distribution smoothing: rare log D values (the heavy
                     lower tail down to -12.5) get up-weighted.  This is the
                     regression analogue of SMOTE that does not fabricate rows.
    ``combo``        geometric mean of group_inv and target_lds.
    """
    n = len(df)
    if scheme in ("none", "", None):
        return None
    if scheme == "group_inv":
        counts = df.groupby("extractant_group")["log_D"].transform("size").to_numpy()
        w = 1.0 / np.sqrt(counts)
    elif scheme == "group_inv_full":
        counts = df.groupby("extractant_group")["log_D"].transform("size").to_numpy()
        w = 1.0 / counts
    elif scheme == "metal_inv":
        w = df["sample_weight_inv_metal_freq"].to_numpy(dtype=float)
    elif scheme == "target_lds":
        w = _lds_weights(df["log_D"].to_numpy())
    elif scheme == "combo":
        counts = df.groupby("extractant_group")["log_D"].transform("size").to_numpy()
        w = np.sqrt((1.0 / np.sqrt(counts)) * _lds_weights(df["log_D"].to_numpy()))
    else:
        raise ValueError(f"unknown weight scheme {scheme!r}")
    w = np.asarray(w, dtype=float)
    w[~np.isfinite(w) | (w <= 0)] = np.nanmedian(w[np.isfinite(w) & (w > 0)])
    return w * (n / w.sum())


def _lds_weights(y: np.ndarray, bins: int = 40, alpha: float = 0.6,
                 kernel_sigma: float = 2.0) -> np.ndarray:
    """Label-distribution-smoothing weights: w ~ 1 / smoothed_density(y)^alpha."""
    hist, edges = np.histogram(y, bins=bins)
    half = int(3 * kernel_sigma)
    ker = np.exp(-0.5 * (np.arange(-half, half + 1) / kernel_sigma) ** 2)
    ker /= ker.sum()
    smooth = np.convolve(hist.astype(float), ker, mode="same")
    smooth = np.maximum(smooth, 1e-6)
    idx = np.clip(np.digitize(y, edges[1:-1]), 0, bins - 1)
    w = (1.0 / smooth[idx]) ** alpha
    return w / w.mean()


# ---------------------------------------------------------------------------
# Target transforms
# ---------------------------------------------------------------------------
class TargetShiftedRegressor(BaseEstimator, RegressorMixin):
    """Optionally learn the residual from a per-extractant offset.

    Not used for the leave-extractants-out headline (an unseen extractant has
    no offset), but useful as a diagnostic upper bound on the within component.
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, y, sample_weight=None):
        self.base_ = clone(self.base)
        try:
            self.base_.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            self.base_.fit(X, y)
        return self

    def predict(self, X):
        return self.base_.predict(X)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def _dense_pipeline(estimator, quantile: bool = False) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=False))]
    steps.append(("scale", QuantileTransformer(output_distribution="normal",
                                               n_quantiles=500, subsample=100000,
                                               random_state=0)
                  if quantile else StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def make_model(name: str, params: dict[str, Any] | None = None, seed: int = 0,
               n_jobs: int = 4):
    """Instantiate a model by name with an optional hyperparameter dict."""
    p = dict(params or {})

    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 1200),
            learning_rate=p.get("learning_rate", 0.03),
            num_leaves=p.get("num_leaves", 63),
            max_depth=p.get("max_depth", -1),
            min_child_samples=p.get("min_child_samples", 20),
            subsample=p.get("subsample", 0.8),
            subsample_freq=p.get("subsample_freq", 1),
            colsample_bytree=p.get("colsample_bytree", 0.6),
            reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 1.0),
            min_split_gain=p.get("min_split_gain", 0.0),
            max_bin=p.get("max_bin", 255),
            objective=p.get("objective", "regression"),
            random_state=seed, n_jobs=n_jobs, verbosity=-1,
        )
    if name == "lgbm_dart":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            boosting_type="dart", n_estimators=p.get("n_estimators", 800),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 63),
            colsample_bytree=p.get("colsample_bytree", 0.6),
            subsample=p.get("subsample", 0.8),
            reg_lambda=p.get("reg_lambda", 1.0),
            random_state=seed, n_jobs=n_jobs, verbosity=-1,
        )
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 1200),
            learning_rate=p.get("learning_rate", 0.03),
            max_depth=p.get("max_depth", 7),
            min_child_weight=p.get("min_child_weight", 3),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.6),
            colsample_bylevel=p.get("colsample_bylevel", 1.0),
            reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 1.0),
            gamma=p.get("gamma", 0.0),
            tree_method="hist", random_state=seed, n_jobs=n_jobs, verbosity=0,
        )
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=p.get("iterations", 1500),
            learning_rate=p.get("learning_rate", 0.04),
            depth=p.get("depth", 7),
            l2_leaf_reg=p.get("l2_leaf_reg", 3.0),
            rsm=p.get("rsm", 0.6),
            bagging_temperature=p.get("bagging_temperature", 1.0),
            random_seed=seed, verbose=0, allow_writing_files=False,
            thread_count=n_jobs,
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=p.get("max_iter", 800),
            learning_rate=p.get("learning_rate", 0.05),
            max_leaf_nodes=p.get("max_leaf_nodes", 31),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 1.0),
            max_features=p.get("max_features", 0.6),
            random_state=seed,
        )
    if name == "rf":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=p.get("n_estimators", 600),
                max_depth=p.get("max_depth", None),
                min_samples_leaf=p.get("min_samples_leaf", 1),
                max_features=p.get("max_features", 0.3),
                random_state=seed, n_jobs=n_jobs))])
    if name == "extratrees":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=p.get("n_estimators", 800),
                max_depth=p.get("max_depth", None),
                min_samples_leaf=p.get("min_samples_leaf", 1),
                max_features=p.get("max_features", 0.4),
                random_state=seed, n_jobs=n_jobs))])
    if name == "ridge":
        return _dense_pipeline(Ridge(alpha=p.get("alpha", 10.0), random_state=seed))
    if name == "elasticnet":
        return _dense_pipeline(ElasticNet(alpha=p.get("alpha", 0.01),
                                          l1_ratio=p.get("l1_ratio", 0.5),
                                          max_iter=5000, random_state=seed))
    if name == "huber":
        return _dense_pipeline(HuberRegressor(alpha=p.get("alpha", 1e-3),
                                              epsilon=p.get("epsilon", 1.35),
                                              max_iter=500))
    if name == "svr":
        return _dense_pipeline(SVR(C=p.get("C", 10.0), epsilon=p.get("epsilon", 0.1),
                                   gamma=p.get("gamma", "scale")), quantile=True)
    if name == "knn":
        return _dense_pipeline(KNeighborsRegressor(
            n_neighbors=p.get("n_neighbors", 10),
            weights=p.get("weights", "distance"),
            metric=p.get("metric", "minkowski"), n_jobs=n_jobs), quantile=True)
    if name == "mlp":
        hidden = p.get("hidden", (256, 128))
        if isinstance(hidden, str):
            hidden = tuple(int(x) for x in hidden.split("-"))
        return _dense_pipeline(MLPRegressor(
            hidden_layer_sizes=hidden,
            alpha=p.get("alpha", 1e-3),
            learning_rate_init=p.get("learning_rate_init", 1e-3),
            batch_size=p.get("batch_size", 128),
            max_iter=p.get("max_iter", 400),
            early_stopping=True, n_iter_no_change=25, validation_fraction=0.12,
            random_state=seed), quantile=True)
    raise ValueError(f"unknown model {name!r}")


AVAILABLE_MODELS = [m for m, ok in (
    ("lgbm", HAS_LGBM), ("lgbm_dart", HAS_LGBM), ("xgb", HAS_XGB),
    ("catboost", HAS_CATBOOST), ("hgb", True), ("rf", True), ("extratrees", True),
    ("ridge", True), ("elasticnet", True), ("huber", True), ("svr", True),
    ("knn", True), ("mlp", True)) if ok]

# Models that read NaN natively -- the rest sit behind a median imputer.
NATIVE_NAN_MODELS = {"lgbm", "lgbm_dart", "xgb", "catboost", "hgb"}


# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------
def suggest_params(trial, name: str) -> dict[str, Any]:
    if name in ("lgbm", "lgbm_dart"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 3000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 400, 3000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
            "rsm": trial.suggest_float("rsm", 0.1, 1.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        }
    if name == "hgb":
        return {
            "max_iter": trial.suggest_int("max_iter", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 50.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        }
    if name in ("rf", "extratrees"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 20, 32]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 16),
            "max_features": trial.suggest_float("max_features", 0.05, 0.9),
        }
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e4, log=True)}
    if name == "elasticnet":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}
    if name == "huber":
        return {"alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 1.05, 3.0)}
    if name == "svr":
        return {"C": trial.suggest_float("C", 0.1, 500.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 0.5, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True)}
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 2, 40),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["minkowski", "manhattan"])}
    if name == "mlp":
        return {
            "hidden": trial.suggest_categorical(
                "hidden", ["512-256", "256-128", "512-256-128", "256-256-128", "1024-256"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
        }
    raise ValueError(f"no search space for {name!r}")
