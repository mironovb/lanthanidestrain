#!/usr/bin/env python3
"""Does the Vietoris-Rips representation encode *which lanthanide* is present?

Why this is the decisive diagnostic
-----------------------------------
The topology-only SNN reaches R2 = 0.48 **between** extractants but only 0.033
**within** them.  Selectivity is entirely a within-extractant, across-metal
question, so that near-zero is the whole story -- but it admits two very
different explanations:

  (a) the representation does not encode metal identity, or
  (b) it does, and the model cannot exploit it for log D.

These call for opposite responses.  Under (a) no amount of architecture work
helps and the representation itself must change.  Under (b) the information is
there and the modelling is at fault.

This probe separates them by asking the encoder to predict the lanthanide index
directly from topology alone -- a far easier, noise-free target with 4,746
labels.  If the probe succeeds, metal identity *is* recoverable and (b) holds.
If it fails, the VR complexes are effectively metal-blind and (a) holds.

The physics says this is hard: the ionic radius step between adjacent
lanthanides is ~0.013 A, while the VR filtration values are quantised over a
0.58-4.0 A range and single-conformer noise in an M-L distance is ~0.05 A.

A permutation control is run alongside: the same probe on shuffled labels must
collapse to ~0, otherwise the probe is measuring leakage rather than signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from automl import evaluation as ev
from automl.dataset import GROUP_COL
from automl.topo.simplicial_data import SimplicialComplexes, Z_TO_IDX
from automl.topo.train import ComplexCache, build_row_table, run_fold

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/artifacts/topo_runs/metal_probe.json"

GENERIC_LN = Z_TO_IDX[57]          # one shared token for every lanthanide


class MaskedMetalCache:
    """ComplexCache that hides *which* lanthanide the metal is.

    Without this the probe is trivial and proves nothing: the node
    Z-embedding vocabulary gives every lanthanide (57-71) its own token, so a
    model can read the element straight off the metal atom without consulting
    a single distance.  Measured that way the probe scores R2 = 0.9995 -- a
    readout of the label, not evidence about geometry.

    Every metal node is remapped to one shared token here, so the only route to
    the lanthanide's identity is the geometry itself: bond lengths, the
    distance-to-metal feature and the filtration radii.  The is_metal flag is
    untouched, so the model still knows *which* atom is the centre.
    """

    def __init__(self, inner):
        self.inner = inner

    def batch(self, ids):
        b = dict(self.inner.batch(ids))
        z = b["z_idx"].clone()
        metal = b["node_feat"][:, 2] > 0.5          # is_metal column
        z[metal] = GENERIC_LN
        b["z_idx"] = z
        return b


def reference_probe(df, groups, seed: int, folds: int) -> dict:
    """Hand-made geometric summaries on the same task -- the yardstick.

    Eight scalars per complex: mean/min/max/sd of metal-donor distance, donor
    count, median distance-to-metal, and two filtration percentiles.  These use
    no element identity, so they face exactly the same masking the encoder does.
    """
    import numpy as _np
    from sklearn.ensemble import HistGradientBoostingRegressor
    z = _np.load(REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz")
    bids = [str(b) for b in z["build_ids"]]
    ptr, coords = z["node_ptr"], z["coordinates"]
    ismetal, isdon = z["is_metal"], z["is_coord_donor"]
    ef, eptr = z["edge_filtration"], z["edge_ptr"]
    feat = {}
    for k in range(len(ptr) - 1):
        a, b = int(ptr[k]), int(ptr[k + 1])
        c = coords[a:b]
        m = ismetal[a:b].astype(bool)
        d = isdon[a:b].astype(bool)
        if m.sum() != 1 or d.sum() == 0:
            continue
        mi = int(_np.argmax(m))
        dd = _np.linalg.norm(c - c[mi], axis=1)
        don = dd[d]
        e = ef[int(eptr[k]):int(eptr[k + 1])]
        feat[bids[k]] = [don.mean(), don.min(), don.max(), don.std(), len(don),
                         _np.median(dd), float(_np.percentile(e, 5)),
                         float(e.mean())]
    f = df["geometry_feature_build_id"].astype(str).map(feat)
    ok = f.notna().to_numpy()
    X = _np.array([v for v in f[ok].tolist()], dtype=float)
    y = df.loc[ok, "lanthanide_index"].to_numpy(float)
    g = groups[ok]
    oof = _np.full(len(y), _np.nan)
    for tr, te in ev.grouped_folds(g, folds, seed=seed):
        mo = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        mo.fit(X[tr], y[tr])
        oof[te] = mo.predict(X[te])
    return {"r2": ev._r2(y, oof), "mae": float(_np.mean(_np.abs(y - oof))),
            "n": int(len(y))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="lanthanide_index")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--filtration-max", type=float, default=3.5)
    ap.add_argument("--heavy-only", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leak-metal-element", action="store_true",
                    help="do NOT mask the metal's element token -- makes the "
                         "probe trivially solvable; for demonstration only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, X, _ = build_row_table("baseline_2d", "snn")
    # Topology only: a probe that could see the tabular block would trivially
    # read the metal off the one-hot metal columns and prove nothing.
    X = X[:, :0]
    S = SimplicialComplexes(verbose=False)
    cache = ComplexCache(S, args.filtration_max, args.heavy_only, device)
    if not args.leak_metal_element:
        cache = MaskedMetalCache(cache)

    cfg = {"dim": args.dim, "layers": args.layers, "dropout": 0.1,
           "head_hidden": 256, "lr": 2e-3, "weight_decay": 1e-4,
           "epochs": args.epochs, "batch_rows": 64, "eval_batch": 128,
           "val_every": 2, "patience": 6, "arch": "snn"}

    groups = df[GROUP_COL].to_numpy()
    y_true = df[args.target].to_numpy(dtype=float)

    results = {}
    for label, shuffled in (("real", False), ("shuffled_control", True)):
        d = df.copy()
        if shuffled:
            rng = np.random.default_rng(0)
            # Shuffle *within* the row table, breaking the structure->metal
            # link while preserving the marginal distribution.
            d[args.target] = rng.permutation(d[args.target].to_numpy())
        d["__y"] = d[args.target].astype(float)

        oof = np.full(len(d), np.nan)
        for k, (tr, te) in enumerate(ev.grouped_folds(groups, args.folds,
                                                      seed=args.seed)):
            oof[te] = run_fold(d, X, cache, tr, te, cfg=cfg, device=device,
                               seed=args.seed, target_col="__y")
            print(f"  [{label}] fold {k}: n_te={len(te)}", flush=True)

        y = d["__y"].to_numpy(dtype=float)
        r2 = ev._r2(y, oof)
        mae = float(np.mean(np.abs(y - oof)))
        # "within 1 lanthanide" is the operationally meaningful accuracy: a
        # probe that lands on the right element or its neighbour is resolving
        # the contraction; one that only gets the light/heavy half is not.
        acc1 = float(np.mean(np.abs(y - oof) <= 1.0))
        results[label] = {"r2": r2, "mae": mae, "within_1_index": acc1}
        print(f"[{label}] R2={r2:+.4f}  MAE={mae:.3f} index units  "
              f"within +/-1 index = {acc1:.3f}", flush=True)

    real, ctrl = results["real"], results["shuffled_control"]

    # Reference model on the SAME task, folds and masking.
    #
    # Without this the probe cannot tell "the representation lacks metal
    # information" from "this encoder fails to extract it" -- and it will
    # confidently print the former.  An earlier version did exactly that,
    # reporting METAL-BLIND at R2 = 0.016 while eight hand-made distance
    # summaries reached R2 = 0.57 on identical folds.  The verdict is only
    # meaningful relative to a reference.
    ref = reference_probe(df, groups, args.seed, args.folds)
    results["reference_gbm_scalars"] = ref
    print(f"[reference] 8 hand-made M-L/filtration scalars, same folds: "
          f"R2={ref['r2']:+.4f}  MAE={ref['mae']:.3f}")

    print("")
    print(f"probe R2  encoder={real['r2']:+.4f}  "
          f"shuffled control={ctrl['r2']:+.4f}  "
          f"reference scalars={ref['r2']:+.4f}")
    if ctrl["r2"] > 0.05:
        print("  !! control did not collapse -- the probe is leaking; "
              "do not interpret the real number")
    elif ref["r2"] < 0.15:
        print("  VERDICT: neither the encoder nor simple geometric summaries")
        print("           recover the metal.  The representation itself is")
        print("           the bottleneck.")
    elif real["r2"] < 0.5 * ref["r2"]:
        print("  VERDICT: the representation DOES carry metal identity")
        print(f"           (reference reaches R2={ref['r2']:+.3f}), but this")
        print(f"           encoder recovers only R2={real['r2']:+.3f}.")
        print("           This is an ARCHITECTURE limitation, not a data one.")
    else:
        print("  VERDICT: the encoder extracts the geometric metal signal")
        print("           about as well as simple summaries do.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = (OUT if not args.leak_metal_element
                else OUT.with_name("metal_probe_leaky_control.json"))
    globals()["OUT"] = out_path
    out_path.write_text(json.dumps({"target": args.target, "results": results,
                               "config": vars(args)}, indent=2))
    print(f"\n[metal-probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
