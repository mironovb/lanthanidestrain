"""CAMPAIGN6 endpoint: combine the surviving arms and take the one held-out look.

Two things happen here, and keeping them apart is the point.

**The stack.** Weights are fitted on PAIR vectors, not rows, nested per
held-out extractant.  ``STACK_FITTING_RESULTS.md`` measured why: a row-fitted
meta-learner gives 79.4 % of its weight to CatBoost, which is the best arm on
levels (+0.4987) and the worst on pairs (+0.1441), and 0.000 to the best pair
arm.  Fitting on the quantity being scored was worth +0.0559 on a held-out half
and was called null only by a 30-look Bonferroni accumulated across four
unrelated campaigns.

**The look.** ``--report`` scores on the c6_split report third, which took no
part in any ranking decision.  Every screening and shortlist number in this
campaign came from the other two thirds, so this is a single pre-declared look
and needs no multiplicity correction.  It is also the only number in the
campaign that should be quoted without the word "exploratory".

    python3 -m automl.topo.c6_final --dirs topo_c6_confirm --arms auto
    python3 -m automl.topo.c6_final --dirs topo_c6_confirm --report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.compare_arms import attach_meta
from automl.topo.lift_report import ensemble, load_dirs

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts"
REPORTS = REPO / "automl/reports"
SPLIT = ART / "c6_split"
FCNN = REPORTS / "oof_fcnn_std_scaler_ens16.parquet"


def _nnls(A: np.ndarray, b: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Non-negative least squares by projected gradient (no scipy).

    Non-negative because a negative weight means the meta-learner is exploiting
    a sign flip it cannot justify chemically, which is how a stack overfits a
    small pair set.  Lifted from pair_stack_probe so the two agree exactly.
    """
    n = A.shape[1]
    w = np.full(n, 1.0 / n)
    G = A.T @ A + ridge * np.eye(n)
    c = A.T @ b
    lr = 1.0 / (np.linalg.eigvalsh(G).max() + 1e-12)
    for _ in range(5000):
        w = np.maximum(0.0, w - lr * (G @ w - c))
    s = w.sum()
    return w / s if s > 1e-9 else np.full(n, 1.0 / n)


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return {k: v.loc[idx] for k, v in frames.items()}


def pair_matrix(frames: dict[str, pd.DataFrame], names: list[str],
                key: str = "composition_key"):
    """dy plus one dp column per arm, and the extractant behind each pair.

    Built through ``evaluation.adjacent_pair_arrays`` per extractant, so the
    pair definition is the metric's own and the cluster label comes free.
    Enumerating pairs by hand here is exactly the mistake evaluation.py:181-192
    records inverting a published result.
    """
    ref = frames[names[0]]
    y = ref["y"].to_numpy(float)
    comp, li = ref[key].to_numpy(), ref["lanthanide_index"].to_numpy()
    g = ref["extractant_group"].to_numpy()
    dy_all, grp_all = [], []
    cols: dict[str, list] = {n: [] for n in names}
    for grp in pd.unique(g):
        m = g == grp
        dy, _ = ev.adjacent_pair_arrays(y[m], y[m], comp[m], li[m])
        if not len(dy):
            continue
        dy_all.append(dy)
        grp_all.append(np.repeat(grp, len(dy)))
        for n in names:
            _, dp = ev.adjacent_pair_arrays(
                y[m], frames[n]["oof"].to_numpy(float)[m], comp[m], li[m])
            cols[n].append(dp)
    dy = np.concatenate(dy_all)
    A = np.column_stack([np.concatenate(cols[n]) for n in names])
    return dy, A, np.concatenate(grp_all)


def nested_pair_stack(frames, names, key="composition_key"):
    """Leave-one-extractant-out NNLS on pair vectors."""
    dy, A, grp = pair_matrix(frames, names, key)
    pred = np.zeros_like(dy)
    weights = {}
    for gtest in pd.unique(grp):
        te = grp == gtest
        tr = ~te
        if tr.sum() < 20:
            pred[te] = A[te].mean(axis=1)
            continue
        w = _nnls(A[tr], dy[tr])
        pred[te] = A[te] @ w
        weights[gtest] = w
    W = np.array(list(weights.values())) if weights else np.zeros((0, len(names)))
    return dy, pred, W


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["topo_c6_confirm"])
    ap.add_argument("--arms", nargs="+", default=None,
                    help="tag substrings picking the arms; default = every "
                         "cell in --dirs with >= --min-seeds")
    ap.add_argument("--min-seeds", type=int, default=8)
    ap.add_argument("--key", default="composition_key",
                    choices=("composition_key", "strict_composition_key"))
    ap.add_argument("--restrict", default=None,
                    help="c6_split partition to score on")
    ap.add_argument("--report", action="store_true",
                    help="shorthand for --restrict report: THE single look")
    ap.add_argument("--partners", nargs="+", default=None,
                    help="substrings selecting which c6_partners *_full arms "
                         "enter the blend; unset = all of them, the published "
                         "behaviour. Nine near-duplicate CatBoost variants now "
                         "sit on disk, so an unfiltered blend is an 11-arm "
                         "stack of correlated models.")
    ap.add_argument("--csv", default=str(REPORTS / "c6_final.csv"))
    args = ap.parse_args()

    partners = args.partners
    restrict = "report" if args.report else args.restrict
    groups = None
    if restrict:
        groups = set((SPLIT / f"{restrict}_extractants.txt").read_text().split())

    cells = load_dirs(args.dirs)
    frames: dict[str, pd.DataFrame] = {}
    for k, slot in cells.items():
        if len(slot["runs"]) < args.min_seeds:
            continue
        name = sorted(slot["tags"])[0].rsplit("_s", 1)[0]
        if args.arms and not any(a in name for a in args.arms):
            continue
        frames[name] = ensemble(slot["runs"])
    if FCNN.exists():
        frames["fcnn_repaired"] = attach_meta(
            pd.read_parquet(FCNN).drop_duplicates("safe_exp_id")
            .set_index("safe_exp_id"))
    # ONLY the full-data partner runs.  A screen+select-restricted parquet has
    # no rows on the report third, and align() intersects indices across every
    # arm -- so including one would silently empty the endpoint.
    # --partners restricts WHICH cpu partners enter the blend.  Unset keeps every
    # one, which is the published behaviour -- but 11 *_full parquets now sit on
    # disk (nine CatBoost variants alone), so an unfiltered run silently builds
    # an 11-arm stack out of near-duplicate models rather than the intended
    # 3-arm one, and reads as topology contributing nothing against a wall of
    # correlated partners.
    for p in sorted((ART / "c6_partners").glob("oof_c6p_*_full.parquet")):
        if partners is not None and not any(w in p.stem for w in partners):
            continue
        frames[p.stem.replace("oof_c6p_", "cpu_").replace("_full", "")] = attach_meta(
            pd.read_parquet(p).drop_duplicates("safe_exp_id")
            .set_index("safe_exp_id"))
    if len(frames) < 2:
        print(f"[final] only {len(frames)} arm(s) available; need >= 2")
        return 1

    frames = align(frames)
    if groups:
        frames = {k: v[v["extractant_group"].isin(groups)]
                  for k, v in frames.items()}
        print(f"[final] scoring on {restrict}: "
              f"{len(next(iter(frames.values())))} rows")

    names = sorted(frames)
    singles = []
    for n in names:
        f = frames[n]
        dy, dp = ev.adjacent_pair_arrays(
            f["y"].to_numpy(float), f["oof"].to_numpy(float),
            f[args.key].to_numpy(), f["lanthanide_index"].to_numpy())
        singles.append({"arm": n, "adj_r2": ev._r2(dy, dp),
                        "logD_r2": ev._r2(f["y"].to_numpy(float),
                                          f["oof"].to_numpy(float)),
                        "n_pairs": len(dy)})
    S = pd.DataFrame(singles).sort_values("adj_r2", ascending=False)
    print("\n--- single arms ---")
    print(S.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    dy, pred, W = nested_pair_stack(frames, names, key=args.key)
    r_stack = ev._r2(dy, pred)
    best = float(S["adj_r2"].max())
    print(f"\n--- nested pair-fitted stack over {len(names)} arms ---")
    print(f"  best single arm          {best:+.4f}  ({S.iloc[0]['arm']})")
    print(f"  pair-fitted nested stack {r_stack:+.4f}"
          f"   ({r_stack - best:+.4f} vs best single)")
    if len(W):
        mw = W.mean(axis=0)
        print("  mean weights: " + ", ".join(
            f"{n}={w:.2f}" for n, w in sorted(zip(names, mw),
                                              key=lambda t: -t[1])[:8]))
    print(f"  pairs scored: {len(dy)}")

    if args.report:
        print("\n[final] THIS IS THE PRE-DECLARED LOOK. The report extractants "
              "took no part in any ranking, so no multiplicity correction "
              "applies -- and no further cell may be selected on this number.")
    else:
        print(f"\n[final] EXPLORATORY (restrict={restrict}). Not the endpoint.")

    S["stack_adj_r2"] = r_stack
    S["restrict"] = restrict or "full"
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    S.to_csv(args.csv, index=False)
    print(f"[final] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
