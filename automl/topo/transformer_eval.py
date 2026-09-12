#!/usr/bin/env python3
"""Evaluate the transformer experiments under the standard protocol.

Three transformer uses, all scored out-of-fold, leave-extractants-out, on
adjacent-pair log SF R^2 -- standalone and as shape sources inside the
anchored system (anchor + tabular shape from the CatBoost 8-seed ensemble):

  attn      attention-based 3D encoder (dense; and the <= cutoff sparse control)
  ft        FT-Transformer level/shape models (tabular)
  blocktf   set transformer over the rows of a block (shape only)
  celltf    set transformer over the (block, metal) cells (shape only)

Modes
  (default)   legacy-population analysis: standalone scores, blends with
              nested (equal-extractant) weights, two-encoder mixes with the
              distance encoder, pair-level correlations, seed halves.
  --freeze    write automl/reports/transformer_rule.json: the fixed blend
              weight per source (median nested weight on the legacy set) and
              the primary endpoints for the held-out look.  Run BEFORE the
              expanded-population runs are launched; committed to git.
  --confirm   the one look at the frozen 444 held-out pairs using the frozen
              weights; refuses to run without the rule file.

Writes automl/reports/transformer_eval.json (or _confirm.json).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.fresh_eval import load_fresh
from automl.topo.topo_shape import (GRID, load_cell, nested_1, nested_2,
                                    pair_basis, score)

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts"
ANCH_LEGACY = ART / "anchored_champ/oof_anch_q60_q60_ens8.parquet"
ANCH_HAS3D = ART / "anchored_champ/oof_anch_q60_q60_has3d_ens4.parquet"
RULE = REPO / "automl/reports/transformer_rule.json"
OUT = REPO / "automl/reports/transformer_eval.json"
OUT_CONFIRM = REPO / "automl/reports/transformer_confirm.json"
W_DIST = 0.35   # the fixed weight of the current best system (I15)


# --------------------------------------------------------------------------
def glob_ensemble(pattern: str, col: str = "oof", seeds=None):
    """Average per-seed OOF parquets matching a glob into one frame with
    columns (safe_exp_id, y, oof).  Returns (frame, n_seeds) or (None, 0)."""
    paths = sorted(glob.glob(pattern))
    if seeds is not None:
        paths = [p for p in paths
                 if any(f"_s{s}." in p or f"_s{s}_" in p for s in seeds)]
    if not paths:
        return None, 0
    frames = [pd.read_parquet(p).drop_duplicates("safe_exp_id")
              .set_index("safe_exp_id") for p in paths]
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    out = frames[0].loc[idx, ["y"]].copy()
    out["oof"] = np.mean([f.loc[idx, col].to_numpy() for f in frames], axis=0)
    return out.reset_index(), len(paths)


def per_seed_scores(pattern: str, col: str = "oof") -> list[float]:
    meta = pd.read_parquet(ART / "matrix/matrix.parquet",
                           columns=["safe_exp_id", "composition_key",
                                    "lanthanide_index"])
    vals = []
    for p in sorted(glob.glob(pattern)):
        d = pd.read_parquet(p)
        # train.py parquets already carry the keys; the tabular ones do not
        need = [c for c in ("composition_key", "lanthanide_index")
                if c not in d.columns]
        if need:
            d = d.merge(meta[["safe_exp_id"] + need], on="safe_exp_id")
        dy, dp = ev.adjacent_pair_arrays(d["y"].to_numpy(float),
                                         d[col].to_numpy(float),
                                         d["composition_key"].to_numpy(),
                                         d["lanthanide_index"].to_numpy())
        vals.append(ev._r2(dy, dp))
    return vals


SOURCES = {
    # name: (glob pattern on the legacy population, column)
    "attn":    (str(ART / "topo_tf_attn/oof_tfattn_s*.parquet"), "oof"),
    "attn_sp": (str(ART / "topo_tf_attn/oof_tfattnsp_s*.parquet"), "oof"),
    "ft":      (str(ART / "anchored_ft/oof_ft_anch_s*.parquet"), "shape"),
    "blocktf": (str(ART / "block_tf/oof_blocktf_s*.parquet"), "shape"),
    "celltf":  (str(ART / "cell_tf/oof_celltf_s*.parquet"), "shape"),
}
SOURCES_HAS3D = {
    "attn":    (str(ART / "topo_tf_attn/oof_tfattnh3d_s*.parquet"), "oof"),
    "attn_sp": (str(ART / "topo_tf_attn/oof_tfattnsph3d_s*.parquet"), "oof"),
    "ft":      (str(ART / "anchored_ft/oof_ft_anch_has3d_s*.parquet"), "shape"),
    "blocktf": (str(ART / "block_tf/oof_blocktf_has3d_s*.parquet"), "shape"),
    "celltf":  (str(ART / "cell_tf/oof_celltf_has3d_s*.parquet"), "shape"),
}


def legacy_analysis() -> dict:
    anch = pd.read_parquet(ANCH_LEGACY)
    dist, n_dist = load_cell("topo_c15", "c15_plw4")
    snn, _ = load_cell("topo_c17", "c17_plw4")
    encoders = {"dist": dist, "snn": snn}
    n_seeds = {"dist": n_dist}
    # Seed-count-matched distance reference: the 8 lowest-numbered seeds of
    # the 32 (fixed rule, not a chosen subset), so a transformer source with
    # 8 seeds is compared with an ensemble of the same size.
    c15 = str(ART / "topo_c15/oof_c15_plw4_s*.parquet")
    seeds_all = sorted({int(re.search(r"_s(\d+)_", os.path.basename(q)).group(1))
                        for q in glob.glob(c15)})
    d8, n8 = glob_ensemble(c15, seeds=seeds_all[:8])
    if d8 is not None:
        encoders["dist8"] = d8
        n_seeds["dist8"] = n8
    for name, (pat, col) in SOURCES.items():
        fr, n = glob_ensemble(pat, col)
        if fr is not None:
            encoders[name] = fr
            n_seeds[name] = n
    dy, dps, grp, n_rows = pair_basis(anch, encoders)
    out = {"n_pairs": int(len(dy)), "n_rows": n_rows, "n_seeds": n_seeds,
           "tabular_only": score(dy, dps["tab"])}
    print(f"{n_rows} rows -> {len(dy)} adjacent pairs; sources: "
          + ", ".join(f"{k} x{v}" for k, v in n_seeds.items()))

    # standalone (pair differences of the source alone) and per-seed spread
    out["standalone"] = {}
    for name in encoders:
        rec = score(dy, dps[name])
        if name in SOURCES:
            ps = per_seed_scores(*SOURCES[name])
            if ps:
                rec["per_seed_mean"] = float(np.mean(ps))
                rec["per_seed_sd"] = float(np.std(ps, ddof=1)) if len(ps) > 1 else 0.0
                rec["n_seeds"] = len(ps)
        out["standalone"][name] = rec
    # full-system standalone scores for FT (level+shape) and flat FT
    for cell in ("ft_anch", "ft_flat"):
        fr, n = glob_ensemble(str(ART / f"anchored_ft/oof_{cell}_s*.parquet"))
        if fr is not None:
            meta = pd.read_parquet(ART / "matrix/matrix.parquet",
                                   columns=["safe_exp_id", "composition_key",
                                            "lanthanide_index"])
            d = fr.merge(meta, on="safe_exp_id")
            dyf, dpf = ev.adjacent_pair_arrays(
                d["y"].to_numpy(float), d["oof"].to_numpy(float),
                d["composition_key"].to_numpy(), d["lanthanide_index"].to_numpy())
            out["standalone"][f"{cell}_system"] = {**score(dyf, dpf),
                                                    "n_seeds": n}

    # blends: anchor + tabular shape + one source, nested weight
    out["blend"] = {}
    for name in encoders:
        pred, ws = nested_1(dy, dps["tab"], dps[name], grp)
        out["blend"][name] = {**score(dy, pred),
                              "w_median": float(np.median(ws)),
                              "w_mean": float(np.mean(ws))}
    # two sources at once, with the distance encoder
    out["blend_with_dist"] = {}
    for name in encoders:
        if name in ("dist", "dist8"):
            continue
        pred, ws = nested_2(dy, dps["tab"], dps["dist"], dps[name], grp)
        out["blend_with_dist"][name] = {
            **score(dy, pred),
            "w_dist_mean": float(np.mean([w[0] for w in ws])),
            "w_other_mean": float(np.mean([w[1] for w in ws]))}

    # descriptive weight curves (not nested): pair-weighted R2 and the
    # equal-extractant MSE the nested procedure minimises, on a coarse grid.
    # Shows whether a nested weight of 0.00 is a knife-edge or a real gap.
    wgrid = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]
    exu = pd.unique(grp)
    out["w_curve"] = {"w": wgrid, "sources": {}}
    for name in encoders:
        r2s, eqs = [], []
        for w in wgrid:
            pred = (1 - w) * dps["tab"] + w * dps[name]
            r2s.append(ev._r2(dy, pred))
            eqs.append(float(np.mean([np.mean((dy[grp == g] - pred[grp == g]) ** 2)
                                      for g in exu])))
        out["w_curve"]["sources"][name] = {
            "r2": r2s, "equal_extractant_mse": eqs,
            "disp_ratio": float(np.std(dps[name]) / np.std(dps["tab"]))}

    # pair-level correlations
    names = list(dps)
    M = np.corrcoef([dps[n] for n in names])
    out["corr"] = {"names": names, "matrix": np.round(M, 3).tolist(),
                   "with_tab_residual": {
                       n: float(np.corrcoef(dps[n], dy - dps["tab"])[0, 1])
                       for n in names if n != "tab"}}

    print("  w-curve (pair R2 | equal-extractant MSE) at w = "
          + " ".join(f"{w:.2f}" for w in wgrid))
    for name, rec in out["w_curve"]["sources"].items():
        print(f"    {name:9s} "
              + " ".join(f"{v:+.3f}" for v in rec["r2"]) + " | "
              + " ".join(f"{v:.4f}" for v in rec["equal_extractant_mse"]))
    for name, rec in out["standalone"].items():
        extra = (f"  per-seed {rec['per_seed_mean']:+.4f} ± "
                 f"{rec['per_seed_sd']:.4f} (n={rec['n_seeds']})"
                 if "per_seed_mean" in rec else "")
        print(f"  standalone {name:14s} R2 {rec['r2']:+.4f}{extra}")
    print(f"  tabular-only          R2 {out['tabular_only']['r2']:+.4f}")
    for name, rec in out["blend"].items():
        print(f"  blend +{name:13s} R2 {rec['r2']:+.4f}  P2 {rec['pearson2']:+.4f}"
              f"  w~{rec['w_median']:.2f}")
    for name, rec in out["blend_with_dist"].items():
        print(f"  blend dist+{name:9s} R2 {rec['r2']:+.4f}  "
              f"w_dist {rec['w_dist_mean']:.2f} w_{name} {rec['w_other_mean']:.2f}")
    return out


def freeze_rule(res: dict) -> None:
    rule = {"frozen_from": "legacy 905-pair analysis (transformer_eval.py)",
            "w_dist_fixed": W_DIST, "sources": {}}
    for name in SOURCES:
        if name in res["blend"]:
            w_leg = round(float(res["blend"][name]["w_median"]), 2)
            # 3D encoders are tested by SUBSTITUTION: the attention encoder
            # takes the distance encoder's place at the I15 weight (0.35),
            # everything else unchanged.  A legacy-fitted weight of zero
            # would make the held-out contrast identically zero, which tests
            # nothing.  Tabular shape sources keep the legacy-fitted weight.
            w_fix = W_DIST if name in ("attn", "attn_sp") else w_leg
            rule["sources"][name] = {
                "w_fixed": w_fix, "w_median_legacy": w_leg,
                "test": ("encoder substitution at the I15 weight"
                         if name in ("attn", "attn_sp")
                         else "legacy-fitted nested weight"),
                "primary": "sign of R2(blend, fixed w) - R2(tabular only) "
                           "on the 444 held-out pairs",
                "secondary": "R2(blend source) - R2(blend dist, w=0.35) on "
                             "the same pairs"}
    RULE.write_text(json.dumps(rule, indent=1))
    print(f"rule frozen -> {RULE}: "
          + ", ".join(f"{k} w={v['w_fixed']}" for k, v in rule["sources"].items()))


def score_pairs(df: pd.DataFrame, col: str, which: str, fresh) -> dict:
    dy_all, dp_all = [], []
    for ck, blk in df.groupby("composition_key"):
        blk = blk.groupby("lanthanide_index", as_index=False)[["y", col]].mean()
        idx = blk["lanthanide_index"].to_numpy()
        yv, pv = blk["y"].to_numpy(), blk[col].to_numpy()
        i, j = np.triu_indices(len(blk), k=1)
        adj = np.abs(idx[i] - idx[j]) == 1
        for a, b in zip(i[adj], j[adj]):
            lo, hi = sorted((int(idx[a]), int(idx[b])))
            is_fresh = (str(ck), lo, hi) in fresh
            if which == "fresh" and not is_fresh:
                continue
            if which == "legacy" and is_fresh:
                continue
            dy_all.append(yv[a] - yv[b]); dp_all.append(pv[a] - pv[b])
    dy, dp = np.asarray(dy_all), np.asarray(dp_all)
    return {"n": int(len(dy)), "r2": ev._r2(dy, dp),
            "pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2)
            if np.std(dp) > 0 else float("nan")}


def confirm() -> dict:
    if not RULE.exists():
        raise SystemExit("no frozen rule; run --freeze first (and commit it)")
    rule = json.loads(RULE.read_text())
    fresh = load_fresh()
    anch = pd.read_parquet(ANCH_HAS3D)
    meta = pd.read_parquet(ART / "matrix/matrix.parquet",
                           columns=["safe_exp_id", "composition_key",
                                    "lanthanide_index"])
    dist, n_dist = glob_ensemble(str(ART / "topo_c19/oof_c19_plw4h3d_s*.parquet"))
    base = anch.merge(dist[["safe_exp_id", "oof"]].rename(columns={"oof": "dist"}),
                      on="safe_exp_id").merge(meta, on="safe_exp_id")
    key = pd.Series(base["composition_key"])
    anchor = pd.Series(base["oof"]).groupby(key).transform("mean")
    st = pd.Series(base["oof"]) - anchor
    sd_ = pd.Series(base["dist"]) - pd.Series(base["dist"]).groupby(key).transform("mean")
    base["tab"] = base["oof"]
    base["blend_dist"] = (anchor + (1 - W_DIST) * st + W_DIST * sd_).to_numpy()
    out = {"rule": rule, "sources": {}}
    for name, spec in rule["sources"].items():
        pat, col = SOURCES_HAS3D[name]
        fr, n = glob_ensemble(pat, col)
        if fr is None:
            out["sources"][name] = {"status": "no expanded-population runs"}
            print(f"[{name}] no has3d parquets -- skipped")
            continue
        df = base.merge(fr[["safe_exp_id", "oof"]].rename(columns={"oof": "src"}),
                        on="safe_exp_id")
        k2 = pd.Series(df["composition_key"])
        a2 = pd.Series(df["tab"]).groupby(k2).transform("mean")
        s_t = pd.Series(df["tab"]) - a2
        s_s = pd.Series(df["src"]) - pd.Series(df["src"]).groupby(k2).transform("mean")
        w = spec["w_fixed"]
        df["blend_src"] = (a2 + (1 - w) * s_t + w * s_s).to_numpy()
        rec = {"n_seeds": n, "w_fixed": w}
        for which in ("fresh", "legacy", "all"):
            b = score_pairs(df, "blend_src", which, fresh)
            t = score_pairs(df, "tab", which, fresh)
            d = score_pairs(df, "blend_dist", which, fresh)
            rec[which] = {"blend_src": b, "tabular": t, "blend_dist": d,
                          "primary_contrast": b["r2"] - t["r2"],
                          "vs_dist_contrast": b["r2"] - d["r2"]}
            print(f"[{name}] {which:6s} n={b['n']:4d}  src-blend {b['r2']:+.4f}"
                  f"  tabular {t['r2']:+.4f}  dist-blend {d['r2']:+.4f}  "
                  f"primary {b['r2'] - t['r2']:+.4f}  vs-dist {b['r2'] - d['r2']:+.4f}")
        rec["primary_pass"] = bool(rec["fresh"]["primary_contrast"] > 0)
        print(f"[{name}] PRIMARY {'PASS' if rec['primary_pass'] else 'FAIL'}")
        out["sources"][name] = rec
    OUT_CONFIRM.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_CONFIRM}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    if args.confirm:
        confirm()
        return 0
    res = legacy_analysis()
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}")
    if args.freeze:
        freeze_rule(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
