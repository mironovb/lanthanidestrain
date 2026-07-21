#!/usr/bin/env python3
"""The 2x2 factorial that separates topology from the training objective.

Pre-registered in ``automl/reports/CONTROL_PREREGISTRATION.md`` before any run
was submitted; this module only executes what was written there.

The question
------------
The published study reports adjacent-pair R2 rising from +0.005 to +0.263, but
every one of its 51 runs uses a topological encoder, and the mechanism it
identified -- train the pairwise *contrast* rather than the absolute value -- is
a property of the objective, not of the representation.  So the gain has never
been attributed.  This crosses {no topology, PI-CNN, SNN} with {plain objective,
contrast objective} at the same 16 seeds and reads the attribution off directly.

Cells are identified by their **recorded config**, not by their filename
---------------------------------------------------------------------------
A tag is something a shell script wrote; a config is what the run actually did.
Classifying on ``run_*.json`` means a mislabelled ``--tag`` cannot silently put
a run in the wrong cell, and the seed set of every cell is asserted rather than
assumed.  Directory allowlists are still applied on top, because two runs can
share a config and not be comparable: ``snn_hybrid`` and ``pi_hybrid`` match the
plain-objective cells exactly but predate the MPSN permutation-invariance fix
and the radial readout, so they would confound the objective with source drift.

Every interval comes from ``adjacent_test.paired_adjacent`` -- the same function
behind every published number in this study -- except the interaction, which
needs a difference of differences and is bootstrapped here over the identical
cluster resampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.adjacent_test import adj_r2, paired_adjacent
from automl.topo.compare_arms import attach_meta, collect

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts"
OUT = REPO / "automl/reports/control_factorial.csv"

# The 16 seeds behind the published SNN ensemble.  Every cell must have all of
# them: a cell ensembled over a different seed set is not a matched control.
SEEDS = (7, 11, 23, 37, 42, 51, 67, 83, 211, 223, 233, 241, 251, 263, 271, 281)

CONTRAST = {"pair_loss_weight": 2.0, "select_on": "adjacent"}
PLAIN = {"pair_loss_weight": 0.0, "select_on": "mse"}

# cell -> (arch, objective, head_hidden, allowed directories)
#
# The new cells are restricted to topo_control so that no pre-fix run can drift
# in; S0 and P0 come from the published directories precisely because they ARE
# the published arms and must not be recomputed.
CELLS: dict[str, dict] = {
    "T0":  dict(arch="tabular", obj=CONTRAST, head=256, dirs=("topo_control",)),
    "T0w": dict(arch="tabular", obj=CONTRAST, head=512, dirs=("topo_control",)),
    "T1":  dict(arch="tabular", obj=PLAIN,    head=256, dirs=("topo_control",)),
    "P1":  dict(arch="picnn",   obj=PLAIN,    head=256, dirs=("topo_control",)),
    "P0":  dict(arch="picnn",   obj=CONTRAST, head=256,
                dirs=("topo_adj_seeds", "topo_adjacent", "topo_control")),
    "S1":  dict(arch="snn",     obj=PLAIN,    head=256, dirs=("topo_control",)),
    "S0":  dict(arch="snn",     obj=CONTRAST, head=256,
                dirs=("topo_adj_seeds", "topo_adjacent")),
}

REFS: dict[str, pd.DataFrame] = {}     # published baselines, filled in main()

LABEL = {"T0": "tabular + contrast", "T0w": "tabular + contrast (wide head)",
         "T1": "tabular + plain", "P1": "PI-CNN + plain",
         "P0": "PI-CNN + contrast", "S1": "SNN + plain",
         "S0": "SNN + contrast"}


# ---------------------------------------------------------------------------
def _matches(cfg: dict, spec: dict) -> bool:
    """Config-level cell membership.

    ``pair_loss_weight``/``select_on`` are absent from runs made before those
    flags existed; ``.get`` with the plain-objective default is correct for
    those, and the directory allowlist is what keeps them out where it matters.
    """
    if cfg.get("arch") != spec["arch"]:
        return False
    if cfg.get("topology_only"):
        return False
    if cfg.get("preset") != "baseline_2d":
        return False
    if int(cfg.get("head_hidden", 256)) != spec["head"]:
        return False
    if float(cfg.get("pair_loss_weight") or 0.0) != spec["obj"]["pair_loss_weight"]:
        return False
    if (cfg.get("select_on") or "mse") != spec["obj"]["select_on"]:
        return False
    # Default encoder geometry only; the wide-SNN and filtration sweeps are
    # separate configurations and are not part of this factorial.
    if spec["arch"] == "snn" and (int(cfg.get("dim", 96)) != 96
                                  or int(cfg.get("layers", 3)) != 3):
        return False
    return True


def load_cells(verbose: bool = True) -> dict[str, dict[int, pd.DataFrame]]:
    out: dict[str, dict[int, pd.DataFrame]] = {}
    for cell, spec in CELLS.items():
        found: dict[int, tuple[Path, Path]] = {}
        for d in spec["dirs"]:
            root = ART / d
            if not root.exists():
                continue
            for j in sorted(root.glob("run_*.json")):
                cfg = json.loads(j.read_text()).get("config", {})
                if not _matches(cfg, spec):
                    continue
                seed = int(cfg.get("seed", -1))
                if seed not in SEEDS:
                    continue
                p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
                if not p.exists():
                    continue
                if seed in found:
                    raise RuntimeError(
                        f"cell {cell} seed {seed} matched twice: "
                        f"{found[seed][0].name} and {j.name} -- ambiguous "
                        f"membership would make the ensemble undefined")
                found[seed] = (j, p)
        out[cell] = {s: pd.read_parquet(p).drop_duplicates("safe_exp_id")
                     .set_index("safe_exp_id")
                     for s, (_, p) in sorted(found.items())}
        if verbose:
            missing = sorted(set(SEEDS) - set(found))
            print(f"  {cell:4s} {LABEL[cell]:32s} seeds={len(found):2d}/16"
                  + (f"  MISSING {missing}" if missing else ""))
    return out


def ensemble(members: dict[int, pd.DataFrame]) -> pd.DataFrame | None:
    """Mean out-of-fold prediction over EVERY seed of a cell.

    All seeds, never a subset: choosing which replicates to average on the test
    metric would manufacture the result the factorial exists to test.
    """
    if not members:
        return None
    idx = None
    for df in members.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    stack = np.vstack([members[s].loc[idx, "oof"].to_numpy(float)
                       for s in sorted(members)])
    ens = members[sorted(members)[0]].loc[idx].copy()
    ens["oof"] = stack.mean(axis=0)
    return attach_meta(ens)


# ---------------------------------------------------------------------------
def pairs_by_cluster(d: pd.DataFrame, groups: np.ndarray
                     ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Adjacent-pair (true, predicted) separations, precomputed per extractant.

    Why this is exactly equivalent, and not an approximation.  Two facts, both
    checked rather than assumed:

    * ``composition_key`` is strictly nested inside ``extractant_group`` (552
      blocks, none spanning two extractants), so every block belongs to exactly
      one cluster and the pairs of a resample are the union of the pairs of its
      clusters.
    * ``adjacent_pair_metrics`` groups by composition key and averages per
      metal, so a cluster drawn twice contributes *identically* to a cluster
      drawn once -- verified directly.  The statistic therefore depends only on
      the SET of clusters drawn.

    Given those, precomputing each cluster's pairs once and concatenating the
    distinct drawn clusters reproduces the same number while doing the groupby
    once per arm instead of once per bootstrap draw.  ``_assert_fast_matches``
    checks that claim against the shared metric before any result uses it.
    """
    y = d["y"].to_numpy(float)
    p = d["oof"].to_numpy(float)
    comp = d["composition_key"].to_numpy()
    li = d["lanthanide_index"].to_numpy()
    gcodes, guniq = pd.factorize(groups)
    out = []
    for i in range(len(guniq)):
        r = np.flatnonzero(gcodes == i)
        out.append(ev.adjacent_pair_arrays(y[r], p[r], comp[r], li[r]))
    return out


def _r2_pairs(dy: np.ndarray, dp: np.ndarray) -> float:
    return ev._r2(dy, dp)


def _assert_fast_matches(d: pd.DataFrame, groups: np.ndarray, n_checks: int = 25,
                         seed: int = 12345) -> None:
    """The fast path must agree with the shared metric on random draws.

    A speedup that silently disagreed with ``adjacent_pair_metrics`` would
    reproduce this study's worst bug -- a figure and a table computing the same
    quantity two different ways -- so the equivalence is tested, at full
    precision, before it is used.
    """
    per = pairs_by_cluster(d, groups)
    y = d["y"].to_numpy(float); p = d["oof"].to_numpy(float)
    comp = d["composition_key"].to_numpy(); li = d["lanthanide_index"].to_numpy()
    gcodes, guniq = pd.factorize(groups)
    rows_by_g = [np.flatnonzero(gcodes == i) for i in range(len(guniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_checks):
        pick = rng.integers(0, len(rows_by_g), len(rows_by_g))
        idx = np.concatenate([rows_by_g[i] for i in pick])
        slow = adj_r2(y[idx], p[idx], comp[idx], li[idx])
        sel = np.unique(pick)
        dy = np.concatenate([per[i][0] for i in sel if len(per[i][0])])
        dp = np.concatenate([per[i][1] for i in sel if len(per[i][0])])
        fast = _r2_pairs(dy, dp)
        if not (np.isfinite(slow) and np.isfinite(fast)):
            continue
        if abs(slow - fast) > 1e-9:
            raise AssertionError(
                f"fast adjacent-pair path disagrees with adjacent_pair_metrics: "
                f"{fast!r} vs {slow!r} (delta {fast - slow:.3e})")


def paired_adjacent_fast(a: pd.DataFrame, b: pd.DataFrame, n_boot: int,
                         seed: int = 0) -> dict | None:
    """Same statistic and same resampling as ``paired_adjacent``, precomputed.

    Returns the identical dictionary shape so callers cannot tell them apart.
    """
    common = a.index.intersection(b.index)
    if len(common) < 0.5 * min(len(a), len(b)):
        return None
    a, b = a.loc[common], b.loc[common]
    groups = a["extractant_group"].to_numpy()
    pa, pb = pairs_by_cluster(a, groups), pairs_by_cluster(b, groups)
    n = len(pa)

    def stat(per, sel):
        dy = [per[i][0] for i in sel if len(per[i][0])]
        dp = [per[i][1] for i in sel if len(per[i][0])]
        if not dy:
            return np.nan
        return _r2_pairs(np.concatenate(dy), np.concatenate(dp))

    full = np.arange(n)
    obs_a, obs_b = stat(pa, full), stat(pb, full)
    rng = np.random.default_rng(seed)
    da, db, dd = [], [], []
    for _ in range(n_boot):
        sel = np.unique(rng.integers(0, n, n))
        va, vb = stat(pa, sel), stat(pb, sel)
        if np.isfinite(va) and np.isfinite(vb):
            da.append(va); db.append(vb); dd.append(vb - va)
    if len(dd) < 30:
        return None
    dd = np.array(dd)
    return {"baseline_obs": obs_a, "arm_obs": obs_b,
            "arm_lo": float(np.percentile(db, 5)),
            "arm_hi": float(np.percentile(db, 95)),
            "delta": float(dd.mean()),
            "lo": float(np.percentile(dd, 5)),
            "hi": float(np.percentile(dd, 95)),
            "p_better": float((dd > 0).mean()),
            "n_boot": len(dd)}


def paired_interaction(cells: dict[str, pd.DataFrame], n_boot: int, seed: int = 0):
    """Cluster bootstrap of (S0 - T0) - (S1 - T1).

    Answers a question no two-arm test can: *is topology only worth anything
    once the contrast is trained?*  All four arms are scored on the same
    resampled extractants in each draw, so the four sampling errors cancel the
    way they do in any paired design.
    """
    need = ("S0", "T0", "S1", "T1")
    if any(cells.get(k) is None for k in need):
        return None
    idx = None
    for k in need:
        idx = cells[k].index if idx is None else idx.intersection(cells[k].index)
    ref = cells["S0"].loc[idx]
    y = ref["y"].to_numpy(float)
    comp = ref["composition_key"].to_numpy()
    li = ref["lanthanide_index"].to_numpy()
    p = {k: cells[k].loc[idx, "oof"].to_numpy(float) for k in need}

    gcodes, guniq = pd.factorize(ref["extractant_group"].to_numpy())
    rows_by_g = [np.flatnonzero(gcodes == i) for i in range(len(guniq))]

    def inter(sel):
        return ((adj_r2(y[sel], p["S0"][sel], comp[sel], li[sel])
                 - adj_r2(y[sel], p["T0"][sel], comp[sel], li[sel]))
                - (adj_r2(y[sel], p["S1"][sel], comp[sel], li[sel])
                   - adj_r2(y[sel], p["T1"][sel], comp[sel], li[sel])))

    obs = inter(np.arange(len(ref)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_by_g), len(rows_by_g))
        sel = np.concatenate([rows_by_g[i] for i in pick])
        v = inter(sel)
        if np.isfinite(v):
            draws.append(v)
    if len(draws) < 30:
        return None
    d = np.array(draws)
    return {"obs": float(obs), "mean": float(d.mean()),
            "lo": float(np.percentile(d, 5)), "hi": float(np.percentile(d, 95)),
            "p_positive": float((d > 0).mean()), "n_boot": len(d),
            "n_rows": int(len(ref))}


# ---------------------------------------------------------------------------
# Pre-registered contrast list.  Order is the order it was written in.
CONTRASTS = [
    ("primary",   "T0",   "S0", "does topology add on top of the objective?"),
    ("secondary", "T1",   "P1", "does topology help with no contrast objective?"),
    ("secondary", "T1",   "S1", "same question, SNN encoder"),
    ("descript",  "T1",   "T0", "what the objective buys without topology"),
    ("descript",  "S1",   "S0", "what the objective buys with the SNN"),
    ("descript",  "P1",   "P0", "what the objective buys with the PI-CNN"),
    ("descript",  "T0",   "P0", "topology on top of the objective, PI-CNN"),
    # How much of the published headline was the baseline rather than the model.
    # T1 is the same MLP the study's FCNN is, trained in this harness instead of
    # sklearn defaults; if that alone closes most of the +0.2426, the headline
    # was measuring baseline quality.
    ("baseline",  "FCNN", "T1", "same-harness tabular MLP vs the published FCNN"),
    ("baseline",  "FCNN", "S0", "the published headline, reproduced here"),
    ("baseline",  "CAT",  "T1", "same-harness tabular MLP vs CatBoost"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--allow-partial", action="store_true",
                    help="score cells that do not yet have all 16 seeds "
                         "(for progress checks only -- never for a result)")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="use the per-draw groupby path instead of the "
                         "precomputed one; identical results, ~50x slower")
    args = ap.parse_args()

    print("=== cell membership (by recorded config, not by tag) ===")
    members = load_cells()
    global REFS
    REFS = {k: attach_meta(v) for k, v in collect().items()
            if k.startswith("baseline::")}
    incomplete = [c for c, m in members.items() if len(m) != len(SEEDS)]
    if incomplete and not args.allow_partial:
        print(f"\nincomplete cells: {incomplete}")
        print("Refusing to report: the pre-registration fixes 16 seeds per "
              "cell, and an ensemble over a different seed set is not a "
              "matched control.  Re-run with --allow-partial for a progress "
              "check only.")
        return 1
    if incomplete:
        print(f"\n*** PARTIAL: {incomplete} -- progress check, NOT a result ***")

    ens = {c: ensemble(m) for c, m in members.items()}
    ens["FCNN"] = REFS.get("baseline::mlp::none")
    ens["CAT"] = REFS.get("baseline::catboost::none")

    print("\n=== per-cell ensembles (leave-extractants-out OOF) ===")
    rows = []
    # The two published baselines, on the same rows and the same metric, so the
    # ladder can be read end to end.  Without them the factorial says how the
    # cells rank against each other but not how either relates to the number the
    # study actually reported.
    for lab, key in (("FCNN (published baseline)", "baseline::mlp::none"),
                     ("CatBoost (published baseline)", "baseline::catboost::none")):
        b = REFS.get(key)
        if b is None:
            continue
        y = b["y"].to_numpy(float); p = b["oof"].to_numpy(float)
        a = adj_r2(y, p, b["composition_key"].to_numpy(),
                   b["lanthanide_index"].to_numpy())
        rows.append({"cell": key.split("::")[1], "label": lab, "n_seeds": 1,
                     "n_rows": len(b), "adj_r2": a, "r2_overall": ev._r2(y, p),
                     "single_mean": a, "single_sd": 0.0})
        print(f"  {'--':4s} {lab:32s} adjR2 = {a:+.4f}   R2 = {ev._r2(y, p):+.4f}")
    for c in ("T1", "T0", "T0w", "P1", "P0", "S1", "S0"):
        e = ens.get(c)
        if e is None:
            continue
        y = e["y"].to_numpy(float); p = e["oof"].to_numpy(float)
        a = adj_r2(y, p, e["composition_key"].to_numpy(),
                   e["lanthanide_index"].to_numpy())
        singles = []
        for d in members[c].values():
            dm = attach_meta(d)
            singles.append(adj_r2(dm["y"].to_numpy(float),
                                  dm["oof"].to_numpy(float),
                                  dm["composition_key"].to_numpy(),
                                  dm["lanthanide_index"].to_numpy()))
        rows.append({"cell": c, "label": LABEL[c], "n_seeds": len(members[c]),
                     "n_rows": len(e), "adj_r2": a, "r2_overall": ev._r2(y, p),
                     "single_mean": float(np.mean(singles)),
                     "single_sd": float(np.std(singles))})
        print(f"  {c:4s} {LABEL[c]:32s} adjR2 = {a:+.4f}   "
              f"R2 = {ev._r2(y, p):+.4f}   "
              f"single {np.mean(singles):+.3f}+/-{np.std(singles):.3f}")

    # Pre-registered: the control's value is the BETTER of T0 and T0w, which
    # inflates the control slightly and therefore makes topology's job harder.
    ctl = "T0"
    if ens.get("T0w") is not None and ens.get("T0") is not None:
        a0 = adj_r2(ens["T0"]["y"].to_numpy(float), ens["T0"]["oof"].to_numpy(float),
                    ens["T0"]["composition_key"].to_numpy(),
                    ens["T0"]["lanthanide_index"].to_numpy())
        aw = adj_r2(ens["T0w"]["y"].to_numpy(float), ens["T0w"]["oof"].to_numpy(float),
                    ens["T0w"]["composition_key"].to_numpy(),
                    ens["T0w"]["lanthanide_index"].to_numpy())
        ctl = "T0w" if aw > a0 else "T0"
        print(f"\n  control (pre-registered max of T0/T0w): {ctl} "
              f"(T0 {a0:+.4f}, T0w {aw:+.4f})")
    for r in rows:
        r["is_control"] = (r["cell"] == ctl)

    if args.fast:
        # Prove the shortcut before using it, on a real cell rather than a toy.
        ref = next((ens[c] for c in ("S0", "T0", "T1") if ens.get(c) is not None),
                   None)
        if ref is not None:
            _assert_fast_matches(ref, ref["extractant_group"].to_numpy())
            print("\n[fast path verified: agrees with adjacent_pair_metrics to "
                  "1e-9 on 25 random cluster resamples]")

    print("\n=== pre-registered contrasts (paired cluster bootstrap over extractants) ===")
    out = []
    for kind, base, arm, question in CONTRASTS:
        # Wherever the control cell appears -- as base or as arm -- it is
        # whichever of T0/T0w scored higher.  Resolved in one place: the parity
        # figure once disagreed with the headline because a rule was applied in
        # two, and the figure reads the resolved name back out of the CSV
        # rather than re-deriving it.
        b, a = (ctl if base == "T0" else base), (ctl if arm == "T0" else arm)
        if ens.get(b) is None or ens.get(a) is None:
            continue
        r = (paired_adjacent_fast if args.fast else paired_adjacent)(
            ens[b], ens[a], args.n_boot, seed=0)
        if r is None:
            # paired_adjacent returns None for two unrelated reasons, and
            # reporting only one of them sent me looking for a row-set bug that
            # was really just --n-boot below its floor.
            shared = len(ens[b].index.intersection(ens[a].index))
            why = ("too few rows in common "
                   f"({shared} of {min(len(ens[a]), len(ens[b]))})"
                   if shared < 0.5 * min(len(ens[a]), len(ens[b]))
                   else f"fewer than 30 usable bootstrap draws from "
                        f"--n-boot {args.n_boot}")
            print(f"  {a} vs {b}: skipped -- {why}")
            continue
        verdict = ("arm better" if r["lo"] > 0 else
                   "arm worse" if r["hi"] < 0 else "not distinguishable")
        out.append({"kind": kind, "base": b, "arm": a,
                    "base_role": base, "arm_role": arm,
                    "question": question, **r, "verdict": verdict})
        star = "**" if kind == "primary" else "  "
        print(f"{star}{a:4s} - {b:4s}  delta = {r['delta']:+.4f} "
              f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  P = {r['p_better']:.2f}  "
              f"{verdict:20s} | {question}")

    # --- attribution, both orderings ------------------------------------------
    # A waterfall implies an order, and the order is a choice: crediting the
    # objective first gives topology whatever is left over, and vice versa.
    # With the full 2x2 both orderings are computable, so neither has to be
    # asserted -- and their average is the Shapley value for a two-factor game,
    # which is the order-free answer.  The gap between them IS the interaction.
    def _a(cell):
        e = ens.get(cell)
        if e is None:
            return None
        return adj_r2(e["y"].to_numpy(float), e["oof"].to_numpy(float),
                      e["composition_key"].to_numpy(),
                      e["lanthanide_index"].to_numpy())

    v = {k: _a(k) for k in ("FCNN", "T1", "T0", "S1", "S0")}
    v["T0"] = _a(ctl)
    if all(v[k] is not None for k in v):
        total = v["S0"] - v["FCNN"]
        harness = v["T1"] - v["FCNN"]
        topo_after_obj = v["S0"] - v["T0"]      # objective credited first
        topo_before_obj = v["S1"] - v["T1"]     # topology credited first
        obj_after_topo = v["S0"] - v["S1"]
        obj_before_topo = v["T0"] - v["T1"]
        shap_topo = 0.5 * (topo_after_obj + topo_before_obj)
        shap_obj = 0.5 * (obj_after_topo + obj_before_topo)
        print(f"\n=== attribution of the published +{total:.4f} "
              f"(SNN ensemble - FCNN) ===")
        print(f"  training the same MLP in this harness  {harness:+.4f}  "
              f"({harness/total:5.1%})")
        print(f"  contrast objective   Shapley           {shap_obj:+.4f}  "
              f"({shap_obj/total:5.1%})   "
              f"[order-dependent range {min(obj_before_topo, obj_after_topo):+.4f} "
              f"to {max(obj_before_topo, obj_after_topo):+.4f}]")
        print(f"  3D topology          Shapley           {shap_topo:+.4f}  "
              f"({shap_topo/total:5.1%})   "
              f"[order-dependent range {min(topo_before_obj, topo_after_obj):+.4f} "
              f"to {max(topo_before_obj, topo_after_obj):+.4f}]")
        print(f"  sum                                    "
              f"{harness + shap_obj + shap_topo:+.4f}  (must equal {total:+.4f})")
        pd.DataFrame([
            {"term": "harness", "value": harness, "share": harness / total,
             "lo_order": harness, "hi_order": harness},
            {"term": "objective", "value": shap_obj, "share": shap_obj / total,
             "lo_order": min(obj_before_topo, obj_after_topo),
             "hi_order": max(obj_before_topo, obj_after_topo)},
            {"term": "topology", "value": shap_topo, "share": shap_topo / total,
             "lo_order": min(topo_before_obj, topo_after_obj),
             "hi_order": max(topo_before_obj, topo_after_obj)},
        ]).to_csv(OUT.with_name("control_attribution.csv"), index=False)

    it = paired_interaction({k: ens.get(k) for k in ("S0", "T0", "S1", "T1")},
                            args.n_boot)
    if it:
        print(f"\n=== interaction (S0-T0) - (S1-T1) ===")
        print(f"  {it['obs']:+.4f}  bootstrap mean {it['mean']:+.4f} "
              f"[{it['lo']:+.3f}, {it['hi']:+.3f}]  "
              f"P(positive) = {it['p_positive']:.2f}  (n={it['n_rows']} rows)")
        print("  positive => topology is worth more once the contrast is trained")
        out.append({"kind": "interaction", "base": "(S1-T1)", "arm": "(S0-T0)",
                    "question": "is topology worth more once the contrast is trained?",
                    "delta": it["mean"], "lo": it["lo"], "hi": it["hi"],
                    "p_better": it["p_positive"], "n_boot": it["n_boot"],
                    "arm_obs": it["obs"],
                    "verdict": ("arm better" if it["lo"] > 0 else
                                "arm worse" if it["hi"] < 0 else
                                "not distinguishable")})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        OUT.with_name("control_cells.csv"), index=False)
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(f"\n[control] wrote {OUT} and {OUT.with_name('control_cells.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
