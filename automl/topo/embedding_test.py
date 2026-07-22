#!/usr/bin/env python3
"""Does the learned topological *representation* help the champion learner?

The 'representations paper' approach the PI originally asked for: use the
simplicial encoder as a learned featuriser and hand its embedding to the model
that is actually best at this task, instead of to the MLP head inside the topo
harness.

Why this is a different question from every arm so far
------------------------------------------------------
Every topological arm scored the SNN *end to end* -- encoder plus the harness's
own MLP head.  That confounds the quality of the representation with the quality
of the head, and there is direct evidence the head is the weaker part: T0w (the
harness MLP on tabular features, contrast objective, 16 seeds) scores +0.2006,
while the repaired sklearn MLP on the *same* features scores +0.2206.  The
harness head is worse than sklearn's on identical inputs.

So the encoder may carry useful information that the head then squanders.  This
tests that directly: CatBoost -- the overall-log D champion, and a learner with
no relationship to the topo harness -- on [tabular | out-of-fold SNN embedding].

Leakage
-------
The embeddings are written by ``train.py --dump-embeddings``, which records each
fold's **test-row** embeddings from a model that never saw those extractants.
The assembled matrix is therefore out-of-fold in exactly the sense the OOF
predictions are.  The downstream CatBoost is then cross-validated on the same
leave-extractants-out folds.  This is cross-fitted stacking; it is the standard
construction and the one the nested-blend analysis already relies on.

Arms (identical rows and folds):

    B0  tabular only                      -- the reference
    B1  tabular + SNN embedding           -- does the representation add?
    B2  tabular + RANDOM-init SNN embedding -- the control.

The control is a randomly initialised encoder, not a tabular one: TabularNet has
a width-zero embedding by construction, so it cannot supply a control vector.  A
random-weights SNN sees the same geometry through the same architecture and has
simply never been trained, which separates "the learned representation carries
signal" from "any projection of the coordinates carries signal".  If trained and
random score alike, the encoder learned nothing worth having.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl import models as mz
from automl.compare import paired_bootstrap
from automl.dataset import BLOCK_PRESETS, GROUP_COL, TARGET
from automl.matrix_cache import load_cache
from automl.topo.adjacent_test import adj_r2
from automl.topo.control_factorial import paired_adjacent_fast

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/embedding_test.csv"


def load_emb(pattern: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean embedding over every seed matching ``pattern``."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    acc, ids = None, None
    for f in files:
        z = np.load(f, allow_pickle=False)
        e = z["embeddings"].astype(np.float64)
        if ids is None:
            ids = np.asarray([str(s) for s in z["safe_exp_id"]])
        acc = e if acc is None else acc + e
    return acc / len(files), ids


def _oof(X, y, groups, *, folds, repeats, seed, n_jobs):
    s = np.zeros(len(y)); c = np.zeros(len(y))
    for rep in range(repeats):
        for tr, te in ev.grouped_folds(groups, n_splits=folds, seed=seed + rep):
            m = mz.make_model("catboost", seed=seed + rep, n_jobs=n_jobs)
            m.fit(X[tr], y[tr])
            s[te] += np.asarray(m.predict(X[te]), float); c[te] += 1
    return s / np.maximum(c, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--emb-dir", default=str(REPO / "automl/artifacts/topo_emb"))
    args = ap.parse_args()

    snn = load_emb(f"{args.emb_dir}/emb_embsnn_*.npz")
    ctl = load_emb(f"{args.emb_dir}/emb_embrand_*.npz")
    if snn is None:
        print(f"no SNN embeddings in {args.emb_dir}; run train.py "
              f"--dump-embeddings first")
        return 1

    df, blocks, _ = load_cache()
    base_cols = blocks.select(BLOCK_PRESETS["baseline_2d"])
    sid = df["safe_exp_id"].astype(str).to_numpy()

    def align(pack):
        e, ids = pack
        pos = {s: i for i, s in enumerate(ids)}
        keep = np.asarray([s in pos for s in sid])
        take = np.asarray([pos[s] for s in sid[keep]])
        return e[take], keep

    Esnn, keep = align(snn)
    sub = df[keep].reset_index(drop=True)
    Xbase = sub[base_cols].to_numpy(np.float32)
    y = sub[TARGET].to_numpy(float)
    groups = sub[GROUP_COL].to_numpy()
    # lanthanide_index is required by paired_adjacent_fast; omitting it crashed
    # the first run *after* it had printed the arm values, so the adjacent-pair
    # contrasts never ran and no CSV was written.
    meta = pd.DataFrame({
        "safe_exp_id": sub["safe_exp_id"].to_numpy(), "y": y,
        "extractant_group": groups,
        "composition_key": sub["composition_key"].to_numpy(),
        "lanthanide_index": sub["lanthanide_index"].to_numpy(),
    }).set_index("safe_exp_id")
    comp = sub["composition_key"].to_numpy(); li = sub["lanthanide_index"].to_numpy()
    print(f"rows={len(sub)}  tabular={Xbase.shape[1]}  emb={Esnn.shape[1]}",
          flush=True)

    arms = {"B0_tabular": Xbase,
            "B1_plus_snn_emb": np.hstack([Xbase, Esnn.astype(np.float32)])}
    if ctl is not None:
        Ectl, keep2 = align(ctl)
        if keep2.sum() == keep.sum():
            arms["B2_plus_random_emb"] = np.hstack(
                [Xbase, Ectl.astype(np.float32)])

    oof = {}
    print("\n=== arms (CatBoost, leave-extractants-out) ===")
    rows = []
    for k, X in arms.items():
        p = _oof(X, y, groups, folds=args.folds, repeats=args.repeats,
                 seed=args.seed, n_jobs=args.n_jobs)
        oof[k] = meta.assign(oof=p)
        a = adj_r2(y, p, comp, li); r = ev._r2(y, p)
        rows.append({"arm": k, "n_feat": X.shape[1], "adj_r2": a, "r2_overall": r})
        print(f"  {k:22s} nfeat={X.shape[1]:5d}  adjR2={a:+.4f}  R2={r:+.4f}",
              flush=True)

    print("\n=== contrasts (paired cluster bootstrap over extractants) ===")
    out = []
    pairs = [("B0_tabular", "B1_plus_snn_emb",
              "does the SNN representation add to CatBoost?")]
    if "B2_plus_random_emb" in arms:
        pairs += [("B0_tabular", "B2_plus_random_emb",
                   "control: does an UNTRAINED encoder's projection add?"),
                  ("B2_plus_random_emb", "B1_plus_snn_emb",
                   "DECISIVE: does TRAINING the encoder add over random?")]
    for a, b, q in pairs:
        res = paired_bootstrap(oof[a], oof[b], n_boot=args.n_boot, seed=0)
        if not res:
            continue
        d, lo, hi, P = res["r2_overall"]
        v = "adds" if lo > 0 else "worse" if hi < 0 else "not distinguishable"
        print(f"  {b} - {a}")
        print(f"    overall  dR2 = {d:+.4f} [{lo:+.4f}, {hi:+.4f}] P={P:.2f}  {v}")
        out.append({"base": a, "arm": b, "question": q, "metric": "r2_overall",
                    "delta": d, "lo": lo, "hi": hi, "p_better": P, "verdict": v})
        # Adjacent-pair as well: it is topology's headline metric, and a
        # representation could help one and not the other -- the water<->octanol
        # block did exactly that, helping overall while hurting selectivity.
        ra = paired_adjacent_fast(oof[a], oof[b], args.n_boot, seed=0)
        if ra:
            va = ("adds" if ra["lo"] > 0 else "worse" if ra["hi"] < 0
                  else "not distinguishable")
            print(f"    adjacent dR2 = {ra['delta']:+.4f} "
                  f"[{ra['lo']:+.4f}, {ra['hi']:+.4f}] P={ra['p_better']:.2f}  {va}")
            out.append({"base": a, "arm": b, "question": q, "metric": "adjacent",
                        "delta": ra["delta"], "lo": ra["lo"], "hi": ra["hi"],
                        "p_better": ra["p_better"], "verdict": va})
    if out:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).to_csv(OUT, index=False)
        pd.DataFrame(rows).to_csv(OUT.with_name("embedding_cells.csv"), index=False)
        print(f"\n[emb] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
