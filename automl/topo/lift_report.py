"""CAMPAIGN6: score every cell, on the metric and on the mechanism.

Cells are identified by their RECORDED CONFIG, never by their tag.  That is the
control_factorial doctrine and it exists because a mistyped ``--tag`` must not
be able to put a run in the wrong cell; here it matters more than usual,
because a campaign cell and a published arm can differ by a single flag.

Two numbers per cell, because the study's own mechanism says an arm earns a
stack slot only if it is BOTH strong on the metric AND decorrelated from its
partner:

  adj_r2      the metric, on the seed ensemble (never a single seed: per-seed
              SD is ~0.047 and an identical config re-runs 0.0092 apart)
  err_corr    correlation of its PAIR errors with the repaired fingerprint
              network's.  Published arms sit at 0.88-0.93; anything lower is
              worth having even if it is individually weaker.

    python3 -m automl.topo.lift_report --dirs topo_c6
    python3 -m automl.topo.lift_report --dirs topo_encoder --min-seeds 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.compare_arms import attach_meta

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts"
REPORTS = REPO / "automl/reports"
REF = REPORTS / "oof_fcnn_std_scaler_ens16.parquet"
SPLIT = ART / "c6_split"

# Everything that makes two runs different ARMS rather than two seeds of one.
# Defaults matter: a run recorded before a flag existed has no key for it, and
# ``.get`` must then return the value that run effectively used.
CELL_KEYS = (
    "arch", "preset", "no_triangles", "topology_only",
    "pair_loss_weight", "select_on", "level_weight", "block_key",
    "block_centre", "film", "pair_head", "pair_head_weight", "pair_reconcile",
    "geometry", "extra_block_mean", "node_angular", "angular_readout",
    "attn_pool", "aux_target", "conformers", "pretrain_epochs",
    "filtration_max", "heavy_only", "dim", "layers", "dropout", "head_hidden",
    "lr", "weight_decay", "epochs", "batch_rows", "folds", "repeats",
    "restrict_groups",
    # campaign 6
    "pair_adj_weight", "pair_adj_only", "pair_loss_kind", "pair_metric_align",
    "radius_slope", "radius_slope_u", "radial_bins", "radial_max",
    "rbf_bins", "rbf_max", "edge_asset", "level_loss", "level_huber_delta",
    "pair_subsample", "level_quantile",
)
DEFAULTS = {
    "pair_adj_weight": 3.0, "pair_adj_only": False, "pair_loss_kind": "sq",
    "pair_metric_align": False, "radius_slope": "off",
    "radius_slope_u": "block", "rbf_bins": None, "rbf_max": None,
    "radial_bins": None, "radial_max": None, "topology_only": False,
    "edge_asset": None, "level_loss": "huber", "level_huber_delta": 1.0,
    "pair_subsample": 1.0, "level_quantile": 0.5,
    "conformers": 1, "pretrain_epochs": 0, "restrict_groups": None,
}


def cell_key(cfg: dict) -> tuple:
    out = []
    for k in CELL_KEYS:
        v = cfg.get(k, DEFAULTS.get(k))
        if k == "restrict_groups" and v:
            v = Path(str(v)).name          # path prefix is machine-specific
        out.append((k, v))
    return tuple(out)


def load_dirs(dirs: list[str], verbose: bool = True) -> dict[tuple, dict]:
    """Group every run under ``dirs`` into cells keyed on recorded config."""
    cells: dict[tuple, dict] = {}
    for d in dirs:
        root = ART / d
        if not root.exists():
            print(f"[lift] no such dir: {root}")
            continue
        for j in sorted(root.glob("run_*.json")):
            rec = json.loads(j.read_text())
            cfg = rec.get("config", {})
            p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
            if not p.exists():
                continue
            k = cell_key(cfg)
            seed = int(cfg.get("seed", -1))
            slot = cells.setdefault(k, {"tags": set(), "runs": {}, "cfg": cfg})
            slot["tags"].add(str(rec.get("tag", "?")))
            if seed in slot["runs"]:
                # Two runs with identical config AND seed: which one is the
                # cell is undefined, and averaging them would quietly halve the
                # seed variance of that one member.
                raise RuntimeError(
                    f"duplicate seed {seed} for one cell: "
                    f"{slot['runs'][seed].name} and {p.name}")
            slot["runs"][seed] = p
    if verbose:
        print(f"[lift] {len(cells)} distinct cells across {dirs}")
    return cells


def ensemble(paths: dict[int, Path]) -> pd.DataFrame:
    """Mean OOF over EVERY seed present -- never a chosen subset."""
    frames = {s: pd.read_parquet(p).drop_duplicates("safe_exp_id")
              .set_index("safe_exp_id") for s, p in sorted(paths.items())}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    stack = np.vstack([frames[s].loc[idx, "oof"].to_numpy(float)
                       for s in sorted(frames)])
    ens = frames[sorted(frames)[0]].loc[idx].copy()
    ens["oof"] = stack.mean(axis=0)
    return attach_meta(ens)


def _restrict(d: pd.DataFrame, groups: set[str] | None) -> pd.DataFrame:
    if not groups:
        return d
    return d[d["extractant_group"].isin(groups)]


def score(cell: pd.DataFrame, ref: pd.DataFrame | None,
          key: str = "composition_key",
          groups: set[str] | None = None) -> dict:
    """Metric, and error-correlation with the reference, on shared rows."""
    a = _restrict(cell, groups)
    out: dict[str, float] = {}
    if not len(a):
        return out
    y = a["y"].to_numpy(float)
    comp, li = a[key].to_numpy(), a["lanthanide_index"].to_numpy()
    dy, dp = ev.adjacent_pair_arrays(y, a["oof"].to_numpy(float), comp, li)
    if not len(dy):
        return out
    out["n_rows"] = len(a)
    out["n_pairs"] = len(dy)
    out["adj_r2"] = ev._r2(dy, dp)

    if ref is None:
        return out
    idx = a.index.intersection(ref.index)
    if len(idx) < 50:
        return out
    aa, bb = a.loc[idx], ref.loc[idx]
    y2 = aa["y"].to_numpy(float)
    c2, l2 = aa[key].to_numpy(), aa["lanthanide_index"].to_numpy()
    dy_a, dp_a = ev.adjacent_pair_arrays(y2, aa["oof"].to_numpy(float), c2, l2)
    dy_b, dp_b = ev.adjacent_pair_arrays(y2, bb["oof"].to_numpy(float), c2, l2)
    # Same rows in the same order => adjacent_pair_arrays yields the same pair
    # ORDER (it iterates blocks in groupby order and metals sorted), so the two
    # dp vectors are element-wise comparable.  Asserted rather than assumed.
    if len(dy_a) != len(dy_b) or not np.allclose(dy_a, dy_b, atol=1e-9):
        return out
    ea, eb = dp_a - dy_a, dp_b - dy_b
    out["ref_adj_r2"] = ev._r2(dy_b, dp_b)
    out["err_corr"] = float(np.corrcoef(ea, eb)[0, 1])
    # An arm that is no better alone but decorrelated is still worth having;
    # this is the cheapest honest indicator of that.
    out["blend50_adj_r2"] = ev._r2(dy_a, 0.5 * (dp_a + dp_b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["topo_c6"])
    ap.add_argument("--min-seeds", type=int, default=3,
                    help="cells with fewer seeds are printed but not ranked")
    ap.add_argument("--key", default="composition_key",
                    choices=("composition_key", "strict_composition_key"))
    ap.add_argument("--restrict", default=None,
                    help="name under automl/artifacts/c6_split (screen, "
                         "select, report, screen_select) to score on")
    ap.add_argument("--baseline", default=None,
                    help="tag substring of the cell to report deltas against")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    groups = None
    if args.restrict:
        p = SPLIT / f"{args.restrict}_extractants.txt"
        groups = set(p.read_text().split())
        print(f"[lift] scoring on {len(groups)} {args.restrict} extractants")

    ref = None
    if REF.exists():
        ref = attach_meta(pd.read_parquet(REF).drop_duplicates("safe_exp_id")
                          .set_index("safe_exp_id"))

    cells = load_dirs(args.dirs)
    rows = []
    for k, slot in cells.items():
        n = len(slot["runs"])
        try:
            ens = ensemble(slot["runs"])
        except Exception as exc:                      # noqa: BLE001
            print(f"[lift] skipping {sorted(slot['tags'])[:1]}: {exc}")
            continue
        rec = {"tags": ",".join(sorted(slot["tags"])[:2]), "n_seeds": n,
               "ranked": n >= args.min_seeds}
        # Only the keys that actually vary across the loaded cells, so the
        # table stays readable when one dir holds one axis.
        rec.update(score(ens, ref, key=args.key, groups=groups))
        for kk, vv in k:
            rec[kk] = vv
        rows.append(rec)

    if not rows:
        print("[lift] nothing to report")
        return 1
    out = pd.DataFrame(rows)
    # Only the config keys that actually vary across the loaded cells, so the
    # table stays readable when one directory holds one axis.
    varying = [c for c in CELL_KEYS
               if c in out.columns and out[c].nunique(dropna=False) > 1]
    show = (["tags", "n_seeds", "n_pairs", "adj_r2", "err_corr",
             "blend50_adj_r2"] + varying)
    show = [c for c in show if c in out.columns]
    out = out.sort_values("adj_r2", ascending=False, na_position="last")

    ranked = out[out["ranked"]]
    if args.baseline is not None and len(ranked):
        base = ranked[ranked["tags"].str.contains(args.baseline, regex=False)]
        if len(base):
            b = float(base.iloc[0]["adj_r2"])
            out["delta"] = out["adj_r2"] - b
            show.insert(4, "delta")
            print(f"[lift] baseline {base.iloc[0]['tags']} adj_r2={b:+.4f}")

    pd.set_option("display.width", 220, "display.max_columns", 60)
    print(out[show].head(args.top).to_string(index=False,
                                             float_format=lambda v: f"{v:+.4f}"))
    n_unranked = int((~out["ranked"]).sum())
    if n_unranked:
        print(f"[lift] {n_unranked} cell(s) below --min-seeds "
              f"{args.min_seeds}: shown, not ranked")
    print(f"[lift] EXPLORATORY -- {len(ranked)} ranked cells is "
          f"{len(ranked)} looks. Nothing here is a confirmed result; the "
          f"shortlist must be re-run on a partition that took no part in the "
          f"ranking.")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.csv, index=False)
        print(f"[lift] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
