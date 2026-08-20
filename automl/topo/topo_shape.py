#!/usr/bin/env python3
"""Does the simplicial encoder (snn) add anything in the shape channel?

Two evaluations, both pure arithmetic on out-of-fold predictions already on
disk (legacy population, 905 adjacent pairs, leave-extractants-out):

1. Encoder comparison in the anchored blend.
   prediction = anchor + (1-w)*shape_tabular + w*shape_encoder, with the
   blend weight chosen nested per held-out extractant (equal-extractant MSE
   criterion, as in anchored_3d.py). Shape sources compared:
     - dist encoder, c15_plw4 32-seed ensemble (current system, reference)
     - snn encoder, c17_plw4 32-seed ensemble (message passing over edges
       AND triangles)
     - both at once: anchor + (1-w1-w2)*shape_tab + w1*shape_dist
       + w2*shape_snn, (w1, w2) on a nested grid.

2. Triangle ablation at matched seeds. Two config-matched snn pairs that
   differ only in --no-triangles, restricted to their 8 shared seeds:
     f4.0: topo_c6_confirm/c6s_s1_g0_f40 (no tri) vs topo_filt/filt4.0 (tri)
     f3.5: topo_encoder/g0_notri (no tri) vs topo_s2_ablate/s2ab_noconf (tri)
   For each side: 8-seed ensemble -> anchored blend -> adjacent R2. The
   tri-vs-notri difference isolates what the 2-simplices contribute.

Writes automl/reports/topo_shape.json.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.topo_shape
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.lift_report import ensemble, load_dirs

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/anchored_champ"
OUT = REPO / "automl/reports/topo_shape.json"

GRID = np.arange(0.0, 1.0001, 0.05)
SHARED_SEEDS = {7, 11, 23, 37, 42, 51, 67, 83}


def load_cell(dirname: str, tag_prefix: str, seed_filter=None) -> pd.DataFrame:
    cells = load_dirs([dirname], verbose=False)
    for k, slot in cells.items():
        name = sorted(slot["tags"])[0].rsplit("_s", 1)[0]
        if name == tag_prefix:
            runs = slot["runs"]
            if seed_filter is not None:
                runs = {s: p for s, p in runs.items() if s in seed_filter}
            if not runs:
                raise SystemExit(f"{dirname}/{tag_prefix}: no seeds after filter")
            return ensemble(runs).reset_index(), len(runs)
    raise SystemExit(f"{tag_prefix} not found in {dirname}")


def pair_basis(anch: pd.DataFrame, encoders: dict[str, pd.DataFrame]):
    """Adjacent-pair arrays: dy, tabular-shape dp, and one dp per encoder."""
    meta = pd.read_parquet(
        REPO / "automl/artifacts/matrix/matrix.parquet",
        columns=["safe_exp_id", "composition_key", "lanthanide_index",
                 "extractant_group"])
    df = anch.rename(columns={"oof": "tab"}).merge(meta, on="safe_exp_id")
    for name, enc in encoders.items():
        df = df.merge(enc[["safe_exp_id", "oof"]].rename(columns={"oof": name}),
                      on="safe_exp_id")
    y = df["y"].to_numpy(float)
    comp = df["composition_key"].to_numpy()
    li = df["lanthanide_index"].to_numpy()
    ex = df["extractant_group"].to_numpy()
    key = pd.Series(comp)
    anchor = pd.Series(df["tab"]).groupby(key).transform("mean").to_numpy()
    shapes = {"tab": df["tab"].to_numpy() - anchor}
    for name in encoders:
        s = pd.Series(df[name])
        shapes[name] = (s - s.groupby(key).transform("mean")).to_numpy()
    cols = {n: [] for n in shapes}
    dy_l, ex_l = [], []
    for g in pd.unique(ex):
        m = ex == g
        first = True
        for n, sh in shapes.items():
            dyg, dpg = ev.adjacent_pair_arrays(y[m], anchor[m] + sh[m],
                                               comp[m], li[m])
            if first and len(dyg):
                dy_l.append(dyg)
                ex_l.append(np.repeat(g, len(dyg)))
            first = False
            if len(dyg):
                cols[n].append(dpg)
    dy = np.concatenate(dy_l)
    return (dy, {n: np.concatenate(v) for n, v in cols.items()},
            np.concatenate(ex_l), len(df))


def nested_1(dy, dst, dse, grp):
    """One encoder: nested blend weight, equal-extractant criterion."""
    exu = pd.unique(grp)
    sse = {g: np.array([np.mean((dy[grp == g] - ((1 - w) * dst[grp == g]
                                                 + w * dse[grp == g])) ** 2)
                        for w in GRID]) for g in exu}
    pred = np.zeros_like(dy)
    ws = []
    for g in exu:
        tr = np.array([sse[h] for h in exu if h != g])
        w = GRID[int(np.argmin(tr.mean(axis=0)))]
        ws.append(float(w))
        m = grp == g
        pred[m] = (1 - w) * dst[m] + w * dse[m]
    return pred, ws


def nested_2(dy, dst, d1, d2, grp):
    """Two encoders: nested (w1, w2) grid, w1 + w2 <= 0.7."""
    combos = [(w1, w2) for w1 in GRID for w2 in GRID
              if w1 + w2 <= 0.7 + 1e-9]
    exu = pd.unique(grp)
    sse = {}
    for g in exu:
        m = grp == g
        sse[g] = np.array([np.mean((dy[m] - ((1 - w1 - w2) * dst[m]
                                             + w1 * d1[m] + w2 * d2[m])) ** 2)
                           for w1, w2 in combos])
    pred = np.zeros_like(dy)
    ws = []
    for g in exu:
        tr = np.array([sse[h] for h in exu if h != g])
        w1, w2 = combos[int(np.argmin(tr.mean(axis=0)))]
        ws.append((float(w1), float(w2)))
        m = grp == g
        pred[m] = (1 - w1 - w2) * dst[m] + w1 * d1[m] + w2 * d2[m]
    return pred, ws


def score(dy, dp):
    return {"r2": ev._r2(dy, dp),
            "pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2)}


def main() -> int:
    anch = pd.read_parquet(ART / "oof_anch_q60_q60_ens8.parquet")
    out = {}

    # --- part 1: dist vs snn vs both in the blend -------------------------
    dist, n_d = load_cell("topo_c15", "c15_plw4")
    snn4, n_s4 = load_cell("topo_c17", "c17_plw4")
    snn2, n_s2 = load_cell("topo_c17", "c17_plw2")
    print(f"ensembles: dist plw4 x{n_d}, snn plw4 x{n_s4}, snn plw2 x{n_s2}")
    dy, dps, grp, n_rows = pair_basis(
        anch, {"dist": dist, "snn4": snn4, "snn2": snn2})
    print(f"{n_rows} rows -> {len(dy)} adjacent pairs")

    res = {"tabular_only": score(dy, dps["tab"])}
    for name in ("dist", "snn4", "snn2"):
        pred, ws = nested_1(dy, dps["tab"], dps[name], grp)
        res[f"blend_{name}"] = {**score(dy, pred),
                                "w_mean": float(np.mean(ws))}
    pred, ws = nested_2(dy, dps["tab"], dps["dist"], dps["snn4"], grp)
    w1m = float(np.mean([w[0] for w in ws]))
    w2m = float(np.mean([w[1] for w in ws]))
    res["blend_dist_plus_snn4"] = {**score(dy, pred),
                                   "w_dist_mean": w1m, "w_snn_mean": w2m}
    out["part1_encoder_comparison"] = res
    for k, v in res.items():
        extra = " ".join(f"{a}={b:.2f}" for a, b in v.items()
                         if a.startswith("w"))
        print(f"  {k:22s} R2={v['r2']:+.4f} P2={v['pearson2']:+.4f} {extra}")

    # --- part 2: triangle ablation at matched seeds -----------------------
    pairs = {
        "f4.0": (("topo_c6_confirm", "c6s_s1_g0_f40"),
                 ("topo_filt", "filt4.0")),
        "f3.5": (("topo_encoder", "g0_notri"),
                 ("topo_s2_ablate", "s2ab_noconf")),
    }
    out["part2_triangle_ablation"] = {}
    for filt, ((d_no, t_no), (d_tri, t_tri)) in pairs.items():
        row = {}
        for label, (d, t) in (("no_triangles", (d_no, t_no)),
                              ("triangles", (d_tri, t_tri))):
            enc, n = load_cell(d, t, seed_filter=SHARED_SEEDS)
            dyh, dph, grph, _ = pair_basis(anch, {"e": enc})
            pred, ws = nested_1(dyh, dph["tab"], dph["e"], grph)
            row[label] = {**score(dyh, pred), "n_seeds": n,
                          "w_mean": float(np.mean(ws)),
                          "encoder_alone": score(dyh, dph["e"])}
        row["blend_r2_tri_minus_notri"] = (row["triangles"]["r2"]
                                           - row["no_triangles"]["r2"])
        out["part2_triangle_ablation"][filt] = row
        print(f"  [{filt}] blend: tri {row['triangles']['r2']:+.4f} vs "
              f"no-tri {row['no_triangles']['r2']:+.4f} "
              f"(diff {row['blend_r2_tri_minus_notri']:+.4f}); "
              f"encoder alone: tri {row['triangles']['encoder_alone']['r2']:+.4f} "
              f"vs no-tri {row['no_triangles']['encoder_alone']['r2']:+.4f}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
