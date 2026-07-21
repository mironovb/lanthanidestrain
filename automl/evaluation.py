#!/usr/bin/env python3
"""Leave-extractants-out protocol and the metric decomposition that matters.

Why decompose R^2
-----------------
With a leave-extractants-out split the model is asked two very different
questions at once:

1. *Between extractants*: is this ligand family a strong or a weak extractant on
   average?  This is largely a ligand-identity question and the 2D fingerprint
   answers it well (baseline R^2_between ~ 0.81).
2. *Within an extractant*: given this ligand, how does log D change from La to
   Lu and from one set of conditions to another?  This is the separation
   problem -- it is what a process chemist actually buys -- and the baseline is
   weak here (R^2_within ~ 0.26).

A single overall R^2 hides that trade-off, so every run reports both, plus a
third, stricter view: within one extractant *and* one fixed condition set, how
well is the lanthanide series ordered (metal-selectivity)?

All metrics are computed from out-of-fold predictions produced by grouped CV on
``extractant_group``: an extractant never appears in both train and test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------
def grouped_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold with genuinely independent repeats.

    ``sklearn.GroupKFold`` without ``shuffle`` balances fold sizes greedily and
    is essentially deterministic given the group sizes -- merely permuting the
    group *labels* leaves ~81 % of each fold unchanged from one seed to the
    next, so "repeats" would silently be near-duplicates and the fold spread
    would look tighter than it is.  sklearn >= 1.6 exposes ``shuffle`` /
    ``random_state``, which randomises the group-to-fold assignment properly;
    fall back to the label permutation only on older sklearn.
    """
    try:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(groups)), groups=groups))
    except TypeError:  # sklearn < 1.6
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        perm = rng.permutation(len(uniq))
        remap = {g: int(perm[i]) for i, g in enumerate(uniq)}
        shuffled = np.array([remap[g] for g in groups])
        return list(GroupKFold(n_splits=n_splits)
                    .split(np.zeros(len(groups)), groups=shuffled))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _r2(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    if len(y) < 2:
        return math.nan
    if w is None:
        w = np.ones(len(y))
    mu = np.average(y, weights=w)
    ss_res = float(np.sum(w * (y - p) ** 2))
    ss_tot = float(np.sum(w * (y - mu) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def variance_decomposed_r2(y: np.ndarray, p: np.ndarray, group: np.ndarray
                           ) -> dict[str, float]:
    """Split R^2 into a between-group and a within-group component.

    between: R^2 of the group means (each group weighted by its size), i.e. can
             the model rank whole extractants?
    within:  R^2 after removing each group's own mean from both y and p, i.e.
             can the model reproduce the spread *inside* an extractant?
    """
    frame = pd.DataFrame({"y": y, "p": p, "g": group})
    gm = frame.groupby("g")[["y", "p"]].transform("mean")
    per_group = frame.groupby("g").agg(y=("y", "mean"), p=("p", "mean"),
                                       n=("y", "size"))
    out = {
        "r2_overall": _r2(frame["y"].to_numpy(), frame["p"].to_numpy()),
        "r2_between": _r2(per_group["y"].to_numpy(), per_group["p"].to_numpy(),
                          per_group["n"].to_numpy(dtype=float)),
        "r2_between_unweighted": _r2(per_group["y"].to_numpy(),
                                     per_group["p"].to_numpy()),
    }
    yc = frame["y"].to_numpy() - gm["y"].to_numpy()
    pc = frame["p"].to_numpy() - gm["p"].to_numpy()
    ss_tot = float(np.sum(yc ** 2))
    out["r2_within"] = 1.0 - float(np.sum((yc - pc) ** 2)) / ss_tot if ss_tot > 0 else math.nan
    out["within_var_frac"] = ss_tot / float(np.sum((frame["y"] - frame["y"].mean()) ** 2))
    # Harsher, per-group view: inside one held-out extractant, does the model beat
    # simply predicting that extractant's own mean?  The pooled `r2_within` above
    # is dominated by the few extractants with a large internal spread; this asks
    # the question separately for every extractant.
    #
    # NOTE ON THE WEIGHTED MEAN: `r2_per_extractant_weighted` is reported for
    # completeness but is NOT a robust statistic -- a group whose own log D
    # barely varies has a near-zero denominator, so its R^2 diverges to large
    # negative values and dominates any average (it reaches about -14 on this
    # dataset).  Quote `r2_per_extractant_median` and
    # `frac_extractants_beat_mean` instead; both are bounded and say the same
    # thing without the tail artefact.
    per = per_group[per_group["n"] >= 10].index
    scores, sizes = [], []
    for g in per:
        sel = frame["g"] == g
        yy = frame.loc[sel, "y"].to_numpy()
        pp = frame.loc[sel, "p"].to_numpy()
        denom = float(np.sum((yy - yy.mean()) ** 2))
        if denom > 0:
            scores.append(1.0 - float(np.sum((yy - pp) ** 2)) / denom)
            sizes.append(len(yy))
    if scores:
        s = np.asarray(scores)
        w = np.asarray(sizes, dtype=float)
        out["r2_per_extractant_weighted"] = float(np.sum(s * w) / w.sum())
        out["r2_per_extractant_median"] = float(np.median(s))
        out["frac_extractants_beat_mean"] = float(np.mean(s > 0))
    if len(yc) > 2 and np.std(yc) > 0 and np.std(pc) > 0:
        out["within_pearson"] = float(np.corrcoef(yc, pc)[0, 1])
        out["within_spearman"] = float(stats.spearmanr(yc, pc).statistic)
    return out


def selectivity_metrics(y: np.ndarray, p: np.ndarray, composition: np.ndarray,
                        metal_index: np.ndarray, min_metals: int = 3
                        ) -> dict[str, float]:
    """Metal-selectivity inside a fixed extractant + fixed condition set.

    For each composition block with >= ``min_metals`` lanthanides measured, we
    ask whether the model orders the series correctly, and how accurate the
    pairwise separation factors log(SF) = log D(A) - log D(B) are.  These are
    the numbers a separation chemist reads off a plot.
    """
    frame = pd.DataFrame({"y": y, "p": p, "c": composition, "m": metal_index})
    rhos, taus, pair_err, pair_true, pair_pred, sign_hits, n_blocks = [], [], [], [], [], [], 0
    for _, blk in frame.groupby("c"):
        blk = blk.groupby("m", as_index=False)[["y", "p"]].mean()
        if len(blk) < min_metals:
            continue
        n_blocks += 1
        if blk["y"].std() > 1e-9 and blk["p"].std() > 1e-9:
            rhos.append(stats.spearmanr(blk["y"], blk["p"]).statistic)
            taus.append(stats.kendalltau(blk["y"], blk["p"]).statistic)
        yv, pv = blk["y"].to_numpy(), blk["p"].to_numpy()
        i, j = np.triu_indices(len(blk), k=1)
        dy, dp = yv[i] - yv[j], pv[i] - pv[j]
        pair_err.extend(np.abs(dy - dp).tolist())
        pair_true.extend(dy.tolist())
        pair_pred.extend(dp.tolist())
        sign_hits.extend((np.sign(dy) == np.sign(dp)).tolist())
    out: dict[str, float] = {"sel_n_blocks": float(n_blocks)}
    if rhos:
        out["sel_spearman_mean"] = float(np.nanmean(rhos))
        out["sel_spearman_median"] = float(np.nanmedian(rhos))
        out["sel_kendall_mean"] = float(np.nanmean(taus))
    if pair_err:
        out["sel_logSF_mae"] = float(np.mean(pair_err))
        out["sel_logSF_r2"] = _r2(np.array(pair_true), np.array(pair_pred))
        out["sel_sign_accuracy"] = float(np.mean(sign_hits))
        out["sel_n_pairs"] = float(len(pair_err))
    out.update(adjacent_pair_metrics(y, p, composition, metal_index))
    return out


def adjacent_pair_arrays(y: np.ndarray, p: np.ndarray, composition: np.ndarray,
                         metal_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True and predicted separations for adjacent lanthanide pairs.

    Split out so that plots and metrics cannot drift apart: a figure that
    enumerates the pairs itself will silently disagree.  That happened once --
    an early parity plot enumerated raw *row* pairs and reported 13,029 pairs
    with the baseline apparently beating the model, exactly inverting the real
    result, because it skipped the within-metal averaging below.

    Replicate measurements of the same metal in the same composition block are
    averaged first, so each (composition, metal) contributes one point and a
    metal measured ten times does not dominate one measured once.
    """
    frame = pd.DataFrame({"y": y, "p": p, "c": composition, "m": metal_index})
    dy_all, dp_all = [], []
    for _, blk in frame.groupby("c"):
        blk = blk.groupby("m", as_index=False)[["y", "p"]].mean()
        if len(blk) < 2:
            continue
        m = blk["m"].to_numpy()
        yv, pv = blk["y"].to_numpy(), blk["p"].to_numpy()
        i, j = np.triu_indices(len(blk), k=1)
        adj = np.abs(m[i] - m[j]) == 1
        if not adj.any():
            continue
        dy_all.extend((yv[i][adj] - yv[j][adj]).tolist())
        dp_all.extend((pv[i][adj] - pv[j][adj]).tolist())
    return np.asarray(dy_all), np.asarray(dp_all)


def adjacent_pair_metrics(y: np.ndarray, p: np.ndarray, composition: np.ndarray,
                          metal_index: np.ndarray) -> dict[str, float]:
    """Separation between *neighbouring* lanthanides -- the hardest case.

    Any model can tell La from Lu; the industrially painful separations are
    between adjacent members of the series, where the ionic radius differs by
    ~0.013 A.  This restricts the pairwise separation-factor statistics to
    pairs with ``|delta lanthanide_index| == 1``, inside a fixed extractant and
    condition block, so it isolates exactly that regime.

    Note Pm is absent from the dataset, so index 5 never occurs and the Nd/Sm
    pair (4 -> 6) is correctly *not* counted as adjacent.
    """
    dy_arr, dp_arr = adjacent_pair_arrays(y, p, composition, metal_index)
    if not len(dy_arr):
        return {}
    hits = (np.sign(dy_arr) == np.sign(dp_arr))
    out = {
        "sel_adj_n_pairs": float(len(dy_arr)),
        "sel_adj_logSF_r2": _r2(dy_arr, dp_arr),
        "sel_adj_logSF_mae": float(np.mean(np.abs(dy_arr - dp_arr))),
        "sel_adj_sign_accuracy": float(np.mean(hits)),
        # How large is the true adjacent-pair separation?  Without this the R2
        # above is easy to over-read: if the true spread is tiny, a good R2 is
        # not a useful model.
        "sel_adj_true_sd": float(np.std(dy_arr)),
    }
    if np.std(dy_arr) > 0 and np.std(dp_arr) > 0:
        out["sel_adj_pearson"] = float(np.corrcoef(dy_arr, dp_arr)[0, 1])
    return out


def full_metrics(y: np.ndarray, p: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    """Every metric for one set of out-of-fold predictions."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out: dict[str, float] = {}
    out.update(variance_decomposed_r2(y, p, meta["extractant_group"].to_numpy()))
    out["rmse"] = float(np.sqrt(np.mean((y - p) ** 2)))
    out["mae"] = float(np.mean(np.abs(y - p)))
    out["medae"] = float(np.median(np.abs(y - p)))
    out["spearman"] = float(stats.spearmanr(y, p).statistic)
    out["pearson"] = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else math.nan
    # Same decomposition but centred on extractant + conditions: what is left is
    # purely the lanthanide series.
    comp = meta["composition_key"].to_numpy()
    frame = pd.DataFrame({"y": y, "p": p, "c": comp})
    gm = frame.groupby("c")[["y", "p"]].transform("mean")
    yc = frame["y"].to_numpy() - gm["y"].to_numpy()
    pc = frame["p"].to_numpy() - gm["p"].to_numpy()
    ss = float(np.sum(yc ** 2))
    out["r2_within_composition"] = (
        1.0 - float(np.sum((yc - pc) ** 2)) / ss if ss > 0 else math.nan
    )
    out.update(selectivity_metrics(y, p, comp, meta["lanthanide_index"].to_numpy()))
    # Per-metal breakdown collapsed to two summary numbers.
    per_metal = pd.DataFrame({"y": y, "p": p, "m": meta["metal"].to_numpy()})
    r2s = per_metal.groupby("m").apply(
        lambda d: _r2(d["y"].to_numpy(), d["p"].to_numpy()), include_groups=False)
    out["r2_per_metal_mean"] = float(np.nanmean(r2s.to_numpy()))
    out["r2_per_metal_min"] = float(np.nanmin(r2s.to_numpy()))
    return out


@dataclass
class CVResult:
    oof: np.ndarray
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]]
    per_fold_r2: list[float]
    extra: dict[str, Any]

    def summary_row(self) -> dict[str, Any]:
        row = dict(self.metrics)
        row["r2_fold_std"] = float(np.std(self.per_fold_r2))
        row["r2_fold_mean"] = float(np.mean(self.per_fold_r2))
        return row


def format_metrics(m: dict[str, float]) -> str:
    keys = ("r2_overall", "r2_between", "r2_within", "r2_within_composition",
            "rmse", "mae", "sel_spearman_mean", "sel_logSF_r2", "sel_sign_accuracy")
    return "  ".join(f"{k.replace('r2_', 'R2_')}={m[k]:.4f}"
                     for k in keys if k in m and np.isfinite(m.get(k, np.nan)))
