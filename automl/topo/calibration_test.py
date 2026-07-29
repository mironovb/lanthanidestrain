#!/usr/bin/env python3
"""Can the magnitude compression be recalibrated away?

The finding this addresses, from ``PUBLICATION_ASSESSMENT.md`` sec 3.3 and the PI
email sec 7: measured adjacent-pair separations span roughly +-2 log units while
predictions span about +-0.5.  Every model "gets direction and ranking
substantially better than magnitude".

Two readings, with opposite consequences:

* **Optimal shrinkage.** Under squared error a model that is genuinely uncertain
  *should* shrink toward the mean; the compression is then a correct response to
  noise, and rescaling would make R2 *worse*.  Nothing to fix, and the paper says
  so.
* **Miscalibration.** The models are trained on absolute log D and the difference
  is a derived quantity, so there is no mechanism forcing the difference to be
  correctly scaled.  A systematic scale error is then free R2, and the paper
  should report the calibrated number.

Which one it is, is a measurement, and it has never been made.  This makes it.

Protocol
--------
The recalibration is fitted **nested by extractant**: the scale for extractant g
comes from the adjacent pairs of the other 161 only, so no pair influences the
transform it is scored under.  Same discipline as the stack weights in
``stack_test.nested_blend``; anything else would be fitting the test set and
would manufacture exactly the improvement being looked for.

Three transforms, from most to least constrained:

* ``scale``   d -> a*d              one parameter; pure magnitude
* ``affine``  d -> a*d + b          adds an offset, which should be ~0 if the
                                    metal ordering convention is symmetric
* ``isotonic`` any monotone map     the most a rank-preserving fix can buy; an
                                    upper bound on what calibration can do at all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.best_stack import nested_stack
from automl.topo.dualkey_test import BINNED, STRICT, load_frames, COMBOS

REPO = Path(__file__).resolve().parents[2]
# Key-specific: the two keys are two different analyses and the first run
# was silently overwritten by the second.
def _out(key: str) -> Path:
    tag = "binned" if key == BINNED else "strict"
    return REPO / f"automl/reports/calibration_test_{tag}.csv"


def _per_group(d: pd.DataFrame, key_col: str):
    """Adjacent-pair (true, predicted) vectors, one entry per extractant."""
    y = d["y"].to_numpy(float)
    p = d["oof"].to_numpy(float)
    comp = d[key_col].to_numpy()
    li = d["lanthanide_index"].to_numpy()
    g = d["extractant_group"].to_numpy()
    out = {}
    for grp in pd.unique(g):
        m = g == grp
        out[grp] = ev.adjacent_pair_arrays(y[m], p[m], comp[m], li[m])
    return out


def _fit_apply(kind: str, dy_tr, dp_tr, dp_te):
    if kind == "scale":
        denom = float(np.sum(dp_tr * dp_tr))
        a = float(np.sum(dy_tr * dp_tr) / denom) if denom > 0 else 1.0
        return a * dp_te, {"a": a, "b": 0.0}
    if kind == "affine":
        A = np.column_stack([dp_tr, np.ones_like(dp_tr)])
        try:
            coef, *_ = np.linalg.lstsq(A, dy_tr, rcond=None)
        except np.linalg.LinAlgError:
            return dp_te, {"a": 1.0, "b": 0.0}
        return coef[0] * dp_te + coef[1], {"a": float(coef[0]),
                                           "b": float(coef[1])}
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(dp_tr, dy_tr)
        return iso.predict(dp_te), {"a": np.nan, "b": np.nan}
    raise ValueError(kind)


def calibrate(per: dict, kind: str):
    """Nested recalibration; returns concatenated (true, calibrated) vectors."""
    groups = [g for g in per if len(per[g][0])]
    dy_all, dp_all, params = [], [], []
    for g in groups:
        others = [o for o in groups if o != g]
        dy_tr = np.concatenate([per[o][0] for o in others])
        dp_tr = np.concatenate([per[o][1] for o in others])
        dy_te, dp_te = per[g]
        cal, pr = _fit_apply(kind, dy_tr, dp_tr, dp_te)
        dy_all.append(dy_te); dp_all.append(cal); params.append(pr)
    return (np.concatenate(dy_all), np.concatenate(dp_all),
            pd.DataFrame(params))


def _summary(dy, dp):
    return {"r2": ev._r2(dy, dp),
            "span_ratio": float(np.std(dp) / np.std(dy)) if np.std(dy) else np.nan,
            "n_pairs": int(len(dy))}


def gain_interval(per: dict, kind: str, n_boot: int = 400, seed: int = 0):
    """Cluster-bootstrap interval on (calibrated R2 - raw R2).

    Without this the gain is a point estimate selected as the best of three
    transforms, which is exactly the shape of claim this study has been caught
    by before: a +0.0178 persistence-image "tuning gain" that replication
    reduced to +0.0003.  A gain smaller than its own interval is not a gain.

    Resamples whole extractants with repetition, the multiplicity-respecting
    form, and applies the *same* draw to both the raw and the calibrated vectors
    so the comparison is paired.
    """
    groups = [g for g in per if len(per[g][0])]
    # Calibrate once, nested, then carry per-extractant vectors into the draws:
    # refitting the transform inside every bootstrap draw would be measuring the
    # transform's own variance, not the gain's.
    cal = {}
    for g in groups:
        others = [o for o in groups if o != g]
        dy_tr = np.concatenate([per[o][0] for o in others])
        dp_tr = np.concatenate([per[o][1] for o in others])
        cal[g] = _fit_apply(kind, dy_tr, dp_tr, per[g][1])[0]

    def stat(pick):
        dy = np.concatenate([per[groups[i]][0] for i in pick])
        raw = np.concatenate([per[groups[i]][1] for i in pick])
        cl = np.concatenate([cal[groups[i]] for i in pick])
        return ev._r2(dy, cl) - ev._r2(dy, raw)

    n = len(groups)
    obs = stat(np.arange(n))
    rng = np.random.default_rng(seed)
    draws = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    draws = np.asarray([d for d in draws if np.isfinite(d)])
    return {"gain": float(obs),
            "lo": float(np.percentile(draws, 5)),
            "hi": float(np.percentile(draws, 95)),
            "p_positive": float((draws > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", default=BINNED, choices=(BINNED, STRICT))
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    frames = load_frames()
    targets = {name: frames[name] for name in ("CatBoost", "repaired", "S0")}
    # the deployed model as well as its parts
    for label, names in COMBOS.items():
        fr, _ = nested_stack(frames, names, key_col=args.key)
        targets[label] = fr

    rows = []
    print(f"=== nested recalibration of the predicted difference "
          f"({args.key}) ===")
    print(f"  {'model':34s} {'raw R2':>8s} {'scale':>8s} {'affine':>8s} "
          f"{'isotonic':>9s}   {'span raw':>9s} {'span cal':>9s}")
    for name, fr in targets.items():
        per = _per_group(fr, args.key)
        raw_dy = np.concatenate([per[g][0] for g in per if len(per[g][0])])
        raw_dp = np.concatenate([per[g][1] for g in per if len(per[g][0])])
        base = _summary(raw_dy, raw_dp)
        res = {"raw": base}
        for kind in ("scale", "affine", "isotonic"):
            dy, dp, pr = calibrate(per, kind)
            res[kind] = _summary(dy, dp)
            res[kind]["median_a"] = float(np.nanmedian(pr["a"])) \
                if "a" in pr else np.nan
        print(f"  {name:34s} {base['r2']:+8.4f} {res['scale']['r2']:+8.4f} "
              f"{res['affine']['r2']:+8.4f} {res['isotonic']['r2']:+9.4f}   "
              f"{base['span_ratio']:9.3f} {res['scale']['span_ratio']:9.3f}")
        for kind, s in res.items():
            rows.append({"key": args.key, "model": name, "transform": kind,
                         **s})

    out = pd.DataFrame(rows)
    dest = _out(args.key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    # The verdict is computed from the numbers, and from their INTERVAL.  The
    # point estimate here is the best of three transforms, which is the same
    # shape of claim as the +0.0178 persistence-image tuning gain that
    # replication reduced to +0.0003.
    key_label = "full (CatBoost+repaired+S0)"
    best = out[out["model"] == key_label]
    if len(best):
        raw = float(best[best["transform"] == "raw"]["r2"].iloc[0])
        span = float(best[best["transform"] == "raw"]["span_ratio"].iloc[0])
        cand = best[best["transform"] != "raw"]
        pick = str(cand.loc[cand["r2"].idxmax(), "transform"])
        per = _per_group(targets[key_label], args.key)
        gi = gain_interval(per, pick, n_boot=args.n_boot)
        print("\n=== verdict ===")
        print(f"  best model raw R2 = {raw:+.4f}; best of three transforms is "
              f"'{pick}'")
        print(f"  gain = {gi['gain']:+.4f} [{gi['lo']:+.4f}, {gi['hi']:+.4f}] "
              f"P(>0)={gi['p_positive']:.2f}   (cluster bootstrap over "
              f"extractants, {args.n_boot} draws)")
        print(f"  predictions span {span:.2f}x the true spread raw, "
              f"{float(cand.loc[cand['r2'].idxmax(), 'span_ratio']):.2f}x after")
        rows.append({"key": args.key, "model": key_label,
                     "transform": f"{pick}_gain", "r2": gi["gain"],
                     "lo": gi["lo"], "hi": gi["hi"],
                     "p_positive": gi["p_positive"]})
        pd.DataFrame(rows).to_csv(dest, index=False)
        if gi["lo"] > 0:
            print("  ==> MISCALIBRATION: a nested rescale improves out-of-sample "
                  "R2 by more than\n      its own interval, so the calibrated "
                  "number is the one to report.")
        else:
            print("  ==> OPTIMAL SHRINKAGE, not miscalibration. The gain does "
                  "not clear its own\n      interval, so rescaling buys nothing "
                  "reliable: the models are already as\n      sharp as the noise "
                  "permits, and the compression is a property of the\n      "
                  "problem rather than a defect to be fixed.")
        print(f"  Recalibration recovers only part of the span "
              f"({span:.2f}x -> "
              f"{float(cand.loc[cand['r2'].idxmax(), 'span_ratio']):.2f}x of the "
              f"true spread),\n  so magnitude compression remains real either "
              f"way and must stay in the paper.")
    print(f"\n[calibration] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
