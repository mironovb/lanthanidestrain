#!/usr/bin/env python3
"""Stage 5b/6: score every topological arm against both baselines, on the rows
and folds they all share.

What this answers, in the abstract's own terms
----------------------------------------------
1. Does 3D topological information improve accuracy over a 2D baseline?
   Reported against **both** baselines -- the FCNN on ECFP + RDKit that the
   abstract benchmarks against, and CatBoost + inverse-extractant weighting,
   which the prior study found is far stronger.  A win over only the weak one
   is reported as exactly that.
2. Are the gains largest for **adjacent lanthanide pairs**?  The adjacent-pair
   metric is scored per arm, so the claim is checked directly rather than
   inferred from the overall number.
3. Do fixed topological descriptors suffice, or is end-to-end learning needed?
   Fixed persistence features -> PI-CNN -> SNN sit on one axis here.

Every delta comes from ``automl.compare.paired_bootstrap``, which resamples
whole extractants (a cluster bootstrap).  Rows within an extractant are not
independent, so a row-level interval would be far too narrow and would call
noise significant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.compare import paired_bootstrap

REPO = Path(__file__).resolve().parents[2]
TOPO_DIR = REPO / "automl/artifacts/topo_runs"
BASE_DIR = REPO / "automl/artifacts/sweeps/topo_baselines"


def _load_oof(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).drop_duplicates("safe_exp_id").set_index("safe_exp_id")


def collect() -> dict[str, pd.DataFrame]:
    """Every arm's OOF predictions, keyed by a readable label."""
    arms: dict[str, pd.DataFrame] = {}
    for p in sorted(TOPO_DIR.glob("oof_*.parquet")):
        arms[p.stem.replace("oof_", "")] = _load_oof(p)
    # The sweep records one JSON line per run in results.jsonl and names the
    # OOF file by a spec hash, so labels have to come from that index rather
    # than from a sibling file per parquet.
    by_hash: dict[str, str] = {}
    for jl in BASE_DIR.rglob("results.jsonl"):
        for line in jl.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            spec = rec.get("spec", {})
            h = str(rec.get("oof") or rec.get("oof_path") or rec.get("hash") or "")
            stem = Path(h).stem if h else ""
            lab = (f"baseline::{spec.get('model','?')}"
                   f"::{spec.get('weight_scheme','none')}")
            if stem:
                by_hash[stem] = lab
            by_hash[rec.get("key", "")] = lab
    for p in sorted(BASE_DIR.rglob("oof_*.parquet")):
        arms[by_hash.get(p.stem, f"baseline::{p.stem}")] = _load_oof(p)
    return arms


REQUIRED_META = ("extractant_group", "composition_key", "metal", "lanthanide_index")


def arm_metrics(d: pd.DataFrame) -> dict[str, float]:
    """Exactly the metric set every other part of the study reports.

    Reusing ``full_metrics`` rather than recomputing R2 here is deliberate: an
    arm scored with a slightly different decomposition would look better or
    worse for reasons that have nothing to do with the model.
    """
    y = d["y"].to_numpy(float)
    p = d["oof"].to_numpy(float)
    out: dict[str, float] = {"n_rows": len(d)}
    missing = [c for c in REQUIRED_META if c not in d.columns]
    if missing:
        # Say so rather than quietly reporting a thinner metric set.
        out["r2_overall"] = ev._r2(y, p)
        out["mae"] = float(np.mean(np.abs(y - p)))
        out["missing_meta"] = ",".join(missing)
        return out
    out.update(ev.full_metrics(y, p, d))
    return out


def attach_meta(d: pd.DataFrame) -> pd.DataFrame:
    """Fill in metadata columns the tabular sweep does not write to its OOF.

    The topological arms write lanthanide_index; the tabular sweep does not.
    Without it the baselines silently drop out of the adjacent-pair comparison
    -- which is the abstract's central claim -- so it is joined back on
    safe_exp_id rather than left missing.
    """
    need = [c for c in REQUIRED_META if c not in d.columns]
    if not need:
        return d
    from automl.matrix_cache import load_cache
    src, _, _ = load_cache()
    cols = ["safe_exp_id"] + [c for c in need if c in src.columns]
    if len(cols) == 1:
        return d
    add = src[cols].drop_duplicates("safe_exp_id").set_index("safe_exp_id")
    return d.join(add, how="left")



def adjacent_pair_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 400,
                            seed: int = 0) -> dict | None:
    """Cluster-bootstrap CI for the adjacent-pair logSF R2 difference.

    ``compare.paired_bootstrap`` covers the R2 decomposition and MAE but not
    the adjacent-pair metric, which is the abstract's headline claim.  Reporting
    that claim from point estimates alone turned out to be untenable: two arms
    differing by 0.006 in overall R2 (snn_hybrid 0.374, snn_wide 0.368) differ
    by 0.265 in adjacent-pair R2 (-0.147 vs +0.118).  A statistic that unstable
    needs an interval attached to every claim made from it.

    Resamples whole extractants, like every other interval in this study.
    """
    common = a.index.intersection(b.index)
    if len(common) < 0.5 * min(len(a), len(b)):
        return None
    a, b = a.loc[common], b.loc[common]
    need = ("composition_key", "lanthanide_index")
    if any(c not in a.columns or c not in b.columns for c in need):
        return None
    groups = a["extractant_group"].to_numpy()
    gcodes, guniq = pd.factorize(groups)
    rows_by_g = [np.flatnonzero(gcodes == i) for i in range(len(guniq))]
    y = a["y"].to_numpy(float)
    pa, pb = a["oof"].to_numpy(float), b["oof"].to_numpy(float)
    comp = a["composition_key"].to_numpy()
    li = a["lanthanide_index"].to_numpy()

    def adj(idx, p):
        m = ev.adjacent_pair_metrics(y[idx], p[idx], comp[idx], li[idx])
        return m.get("sel_adj_logSF_r2", np.nan)

    full = np.arange(len(a))
    obs = {"a": adj(full, pa), "b": adj(full, pb)}
    rng = np.random.default_rng(seed)
    deltas, a_s, b_s = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_by_g), len(rows_by_g))
        idx = np.concatenate([rows_by_g[i] for i in pick])
        va, vb = adj(idx, pa), adj(idx, pb)
        if np.isfinite(va) and np.isfinite(vb):
            a_s.append(va); b_s.append(vb); deltas.append(vb - va)
    if len(deltas) < 50:
        return None
    d = np.array(deltas)
    return {"a_obs": obs["a"], "b_obs": obs["b"],
            "a_lo": float(np.percentile(a_s, 5)), "a_hi": float(np.percentile(a_s, 95)),
            "b_lo": float(np.percentile(b_s, 5)), "b_hi": float(np.percentile(b_s, 95)),
            "delta": float(d.mean()),
            "lo": float(np.percentile(d, 5)), "hi": float(np.percentile(d, 95)),
            "p_better": float((d > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", default=str(REPO / "automl/reports/topo_comparison.csv"))
    args = ap.parse_args()

    arms = collect()
    if not arms:
        print("[compare-arms] no OOF files yet -- nothing to compare")
        return 1

    arms = {k: attach_meta(v) for k, v in arms.items()}
    rows = [{"arm": k, **arm_metrics(v)} for k, v in arms.items()]
    table = pd.DataFrame(rows).sort_values("r2_overall", ascending=False)
    print("\n=== per-arm metrics (leave-extractants-out OOF) ===")
    cols = [c for c in ("arm", "n_rows", "r2_overall", "r2_between", "r2_within",
                        "r2_within_composition", "sel_adj_logSF_r2",
                        "sel_adj_sign_accuracy", "mae") if c in table.columns]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Paired against each baseline separately -- never against only the weak one.
    baselines = [k for k in arms if k.startswith("baseline::")]
    topo = [k for k in arms if not k.startswith("baseline::")]
    deltas = []
    for b in baselines:
        for t in topo:
            res = paired_bootstrap(arms[b], arms[t], n_boot=args.n_boot, seed=0)
            if not res:
                print(f"[compare-arms] {t} vs {b}: rows do not overlap enough "
                      f"to pair -- skipped rather than reported")
                continue
            for metric, (delta, lo, hi, p_better) in res.items():
                deltas.append({"arm": t, "baseline": b, "metric": metric,
                               "delta": delta, "lo": lo, "hi": hi,
                               "p_better": p_better})
    dd = pd.DataFrame(deltas)
    if not dd.empty:
        print("\n=== paired cluster bootstrap (arm minus baseline) ===")
        key = dd[dd["metric"] == "r2_overall"].sort_values("delta", ascending=False)
        for r in key.itertuples():
            if r.lo > 0:
                flag = "   SIGNIFICANT GAIN"
            elif r.hi < 0:
                flag = "   SIGNIFICANT LOSS"
            else:
                flag = "   (interval spans 0)"
            print(f"  {r.arm:24s} vs {r.baseline:34s} "
                  f"dR2 = {r.delta:+.4f}  [{r.lo:+.4f}, {r.hi:+.4f}]  "
                  f"P(better) = {r.p_better:.2f}{flag}")

    # The adjacent-pair claim, with intervals rather than point estimates.
    print("\n=== adjacent-lanthanide-pair logSF R2, cluster-bootstrapped ===")
    ref = "baseline::catboost::none"
    if ref in arms:
        print(f"  (reference = {ref})")
        for t in sorted(arms):
            if t == ref:
                continue
            r = adjacent_pair_bootstrap(arms[ref], arms[t], n_boot=args.n_boot)
            if r is None:
                continue
            span = "spans 0" if r["lo"] <= 0 <= r["hi"] else (
                "excludes 0" if r["hi"] < 0 else "excludes 0 (better)")
            print(f"  {t:38s} adj R2 = {r['b_obs']:+.3f} "
                  f"[{r['b_lo']:+.3f}, {r['b_hi']:+.3f}]   "
                  f"delta vs ref = {r['delta']:+.3f} "
                  f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  {span}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    if not dd.empty:
        dd.to_csv(out.with_name(out.stem + "_paired.csv"), index=False)
    print(f"\n[compare-arms] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
