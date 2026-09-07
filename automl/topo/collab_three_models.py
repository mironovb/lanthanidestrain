#!/usr/bin/env python3
"""The three-model combination scored on the collaborator's expanded
population (5,479 rows, 1,230 adjacent pairs, 177 extractants).

Two of the three arms already exist on that population:
  * flat champion CatBoost   automl/artifacts/anchored_champ/oof_flat_q60_collab_ens4
  * distance encoder         automl/artifacts/topo_c20/oof_c20_plw4col_s*  (4 seeds)
The fingerprint network was never retrained there, so this script runs it
with the repaired recipe (StandardScaler, hidden (256, 128), 16 seeds,
leave-extractants-out 5 x 3) and then fits the pair-level NNLS combination
with c6_final's own machinery.

Writes automl/artifacts/anchored_champ/oof_fcnn_std_ens16_collab.parquet and
automl/reports/collab_three_models.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.dataset import GROUP_COL, TARGET
from automl.topo.anchored_champion import load_table
from automl.topo.c6_final import align, nested_pair_stack
from automl.topo.fcnn_diagnostic import _fit_predict

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/anchored_champ"
OOF = ART / "oof_fcnn_std_ens16_collab.parquet"
OUT = REPO / "automl/reports/collab_three_models.json"
SEEDS = [42, 51, 67, 83, 91, 103, 107, 109, 7, 11, 23, 37, 113, 127, 131, 137]


def adj(frame: pd.DataFrame) -> float:
    return ev.adjacent_pair_metrics(
        frame["y"].to_numpy(float), frame["oof"].to_numpy(float),
        frame["composition_key"].to_numpy(),
        frame["lanthanide_index"].to_numpy())["sel_adj_logSF_r2"]


def run_fcnn(df: pd.DataFrame, X: np.ndarray, seeds: list[int],
             folds: int = 5, repeats: int = 3) -> pd.DataFrame:
    y = df[TARGET].to_numpy(float)
    groups = df[GROUP_COL].to_numpy()
    acc = np.zeros(len(df))
    for k, s in enumerate(seeds, 1):
        t0 = time.time()
        oof_sum, oof_cnt = np.zeros(len(df)), np.zeros(len(df))
        for rep in range(repeats):
            for tr, te in ev.grouped_folds(groups, n_splits=folds,
                                           seed=42 + rep):
                oof_sum[te] += _fit_predict(X, y, tr, te, seed=s,
                                            mode="std_scaler", groups=groups)
                oof_cnt[te] += 1
        acc += oof_sum / np.maximum(oof_cnt, 1)
        run = pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y,
                            "oof": acc / k})
        print(f"  seed {s} ({k}/{len(seeds)}) {time.time()-t0:.0f}s "
              f"· running ens adj R2 = "
              f"{adj(run.join(df[['composition_key','lanthanide_index']])):+.4f}",
              flush=True)
    return pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y,
                         "oof": acc / len(seeds)})


def load_arms(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    meta = df[["safe_exp_id", "composition_key", "lanthanide_index",
               GROUP_COL]].set_index("safe_exp_id")
    frames = {}

    def attach(d: pd.DataFrame, name: str):
        d = d[["safe_exp_id", "y", "oof"]].set_index("safe_exp_id")
        frames[name] = d.join(meta, how="inner")

    attach(pd.read_parquet(ART / "oof_flat_q60_collab_ens4.parquet"), "trees")
    attach(pd.read_parquet(OOF), "fcnn")
    fs = sorted((REPO / "automl/artifacts/topo_c20")
                .glob("oof_c20_plw4col_s*_dist_*.parquet"))
    enc = None
    for i, f in enumerate(fs):
        d = pd.read_parquet(f)[["safe_exp_id", "y", "oof"]]
        enc = d if enc is None else enc.merge(
            d[["safe_exp_id", "oof"]], on="safe_exp_id", suffixes=("", f"_{i}"))
    cols = [c for c in enc.columns if c.startswith("oof")]
    enc["oof"] = enc[cols].mean(axis=1)
    print(f"  encoder ensemble over {len(fs)} seeds")
    attach(enc, "enc3d")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--skip-fcnn", action="store_true")
    a = ap.parse_args()

    df, X, _ = load_table("collab")
    print(f"collab population: {len(df)} rows · {X.shape[1]} columns · "
          f"{df[GROUP_COL].nunique()} extractants", flush=True)

    if not a.skip_fcnn:
        print(f"fingerprint network, {a.seeds} seeds:", flush=True)
        run_fcnn(df, X, SEEDS[:a.seeds]).to_parquet(OOF)
        print(f"wrote {OOF}", flush=True)

    frames = align(load_arms(df))
    names = ["trees", "fcnn", "enc3d"]
    res = {n: float(adj(frames[n])) for n in names}
    dy, pred, W = nested_pair_stack(frames, names)
    res["combined"] = float(ev._r2(dy, pred))
    res["n_pairs"] = int(len(dy))
    res["n_rows"] = int(len(frames[names[0]]))
    res["n_extractants"] = int(df[GROUP_COL].nunique())
    res["stack_weights_mean"] = dict(zip(names, W.mean(axis=0).round(4).tolist()))
    OUT.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:20s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
