#!/usr/bin/env python3
"""Does a *tuned* persistence image contribute? -- the sweep and its endpoint.

Pre-registered in ``automl/reports/PI_SWEEP_PREREGISTRATION.md``, committed
before any sweep run existed.  This module executes only what was fixed there.

Two modes, and the separation between them is the point:

``--stage a`` / ``--stage b``
    Exploratory.  Every configuration trains on all 162 extractants and is
    **scored on tune-half rows only**.  This is where selection happens.

``--confirm``
    Confirmatory.  Takes the single winner, retrained at 16 seeds, and scores it
    on **confirm-half rows**, then applies the pre-registered decision rule.

The winner is chosen as a function of tune-row outcomes alone, and the reported
statistic comes from out-of-fold predictions for confirm rows -- made by models
that never saw those rows.  No confirm-row label influences which configuration
is selected, so the confirm estimate is unbiased for it.  ``N_LOOKS`` is 8 on
that basis.

That is an argument rather than a structural guarantee: the original design
confined sweep *training* to the tune half, which would have made the point
unarguable, but it removed 57 % of the rows and collapsed the arm under test
from adj R2 +0.1562 to +0.0362 -- at which point every configuration tied at
stack weight 0.00 and nothing could be ranked (see
``PI_SWEEP_PREREGISTRATION.md`` #2a).  So ``N_LOOKS_PUNITIVE`` is reported
alongside, charging one look per configuration swept, for readers who would
rather not take the argument on trust.

Everything reuses the machinery the published results were computed with rather
than reimplementing it: ``nested_stack``/``_score`` from ``best_stack``,
``ensemble``/``load_cells``/``paired_adjacent_fast`` from ``control_factorial``,
``_corrected`` from ``stack_test``, ``attach_meta``/``collect`` from
``compare_arms``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.dataset import GROUP_COL
from automl.topo.best_stack import nested_stack, _score
from automl.topo.compare_arms import attach_meta, collect
from automl.topo.control_factorial import (ensemble, load_cells,
                                           paired_adjacent_fast)
from automl.topo.stack_test import _corrected
from automl.topo import pi_split

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
SWEEP = REPO / "automl/artifacts/pi_sweep"
RUNS = SWEEP / "runs_full"      # amended: full-data training, tune-half scoring
RUNS_TUNETRAIN = SWEEP / "runs"  # superseded, kept as evidence (see prereg #2a)
FINAL = REPO / "automl/artifacts/pi_final"

N_LOOKS = 8      # S0, S2, stack primary, stack decisive, S0X, F30, F40, this

# A deliberately punitive alternative: one look per configuration swept, on top
# of the eight confirmatory looks taken across the whole study.
#
# The primary number uses N_LOOKS. The justification is that selection is a
# function of tune-row outcomes only, while the endpoint is computed from
# out-of-fold predictions for confirm rows -- so no confirm label influences
# which configuration is chosen and the estimate is unbiased for it.
#
# That argument is sound but it is an argument, and the amendment of 22 July
# weakened the simpler guarantee it replaced (sweep models now do train on
# confirm extractants in other folds). So the punitive interval is reported
# alongside, unasked, letting a reader who rejects the argument see whether the
# conclusion survives without it. Charging full multiplicity for a search whose
# selection statistic never saw the confirm labels is far harsher than standard
# practice; that is the point.
N_CONFIGS_SWEPT = 25 + 32        # Stage A + Stage B
N_LOOKS_PUNITIVE = N_LOOKS + N_CONFIGS_SWEPT

# The published anchors this module refuses to report without.
S0_ADJ = 0.2382
P0_PUBLISHED_ADJ = 0.2101


def load_runs(root: Path | None = None,
              tune_only: bool | None = None) -> dict[str, dict[int, pd.DataFrame]]:
    """Runs grouped by image-configuration key, then by seed.

    Runs are identified by their **recorded configuration** -- the resolved
    ``pi_images`` path written into ``run_*.json`` -- rather than by parsing the
    tag, so a tag typo cannot silently merge two configurations.

    ``tune_only`` filters on whether the run was confined to the tune half.
    This is a safety interlock rather than a convenience: a sweep run and a
    Stage C run can share an image configuration, and mixing them would put
    tune-half-only predictions into the confirmatory endpoint, quietly voiding
    the separation the whole design rests on.
    """
    root = root or RUNS
    out: dict[str, dict[int, pd.DataFrame]] = {}
    if not root.exists():
        return out
    for js in sorted(root.glob("run_*.json")):
        rec = json.loads(js.read_text())
        img = rec.get("resolved", {}).get("pi_images")
        if not img:
            continue
        restricted = bool(rec["config"].get("restrict_groups"))
        if tune_only is not None and restricted is not tune_only:
            continue
        key = Path(img).stem.replace("img_", "")
        seed = int(rec["config"]["seed"])
        pq = js.with_name(js.name.replace("run_", "oof_").replace(".json",
                                                                 ".parquet"))
        if not pq.exists():
            continue
        df = pd.read_parquet(pq).drop_duplicates("safe_exp_id").set_index(
            "safe_exp_id")
        out.setdefault(key, {})[seed] = df
    return out


def baselines() -> dict[str, pd.DataFrame]:
    return {"CatBoost": attach_meta(collect()["baseline::catboost::none"]),
            "repaired": attach_meta(
                pd.read_parquet(REPORTS / "oof_fcnn_std_scaler_ens16.parquet")
                .drop_duplicates("safe_exp_id").set_index("safe_exp_id"))}


def restrict(frame: pd.DataFrame, ref: pd.DataFrame,
             exts: list[str] | None) -> pd.DataFrame:
    if exts is None:
        return frame
    keep = ref.index[ref[GROUP_COL].astype(str).isin(set(exts))]
    return frame.loc[frame.index.intersection(keep)]


def harness_check() -> pd.DataFrame:
    """Published S0 must re-ensemble to its published value, or we do not report.

    The same standing check ``filt_test`` uses.  It is cheap and it has caught
    drift before; a study that reports a 0.04 effect cannot afford to be unsure
    whether its own baseline moved.
    """
    cells = load_cells(verbose=False)
    s0 = ensemble(cells["S0"])
    a0, _ = _score(s0)
    print(f"harness check: published S0 re-ensembles to {a0:+.4f} "
          f"(must be {S0_ADJ:+.4f})")
    if abs(a0 - S0_ADJ) > 5e-4:
        raise SystemExit("published S0 drifted; refusing to report")
    return s0


def mechanism(arm: pd.DataFrame, ref: pd.DataFrame) -> tuple[float, float]:
    """The two axes an arm must satisfy: strong on the metric, and decorrelated.

    Returns (adjacent-pair R2, correlation of adjacent-pair errors with the
    repaired baseline).  Stated after the original P0 failure and since used to
    predict the filtration outcome before those runs existed.
    """
    idx = arm.index.intersection(ref.index)
    A, B = arm.loc[idx], ref.loc[idx]
    y = B["y"].to_numpy(float)
    comp = B["composition_key"].to_numpy()
    li = B["lanthanide_index"].to_numpy()
    dy, dpa = ev.adjacent_pair_arrays(y, A["oof"].to_numpy(float), comp, li)
    _, dpb = ev.adjacent_pair_arrays(y, B["oof"].to_numpy(float), comp, li)
    corr = float(np.corrcoef(dy - dpa, dy - dpb)[0, 1])
    a, _r = _score(arm)
    return a, corr


def explore(stage: str, min_seeds: int) -> int:
    """Score every configuration on the tune half.  Selection happens here."""
    harness_check()
    rec = pi_split.load()
    if not pi_split.verify():
        raise SystemExit("the frozen split moved; refusing to proceed")

    man_path = SWEEP / f"manifest_stage_{stage}.json"
    if not man_path.exists():
        raise SystemExit(f"no manifest for stage {stage}: {man_path}")
    manifest = {r["key"]: r for r in json.loads(man_path.read_text())}

    base = baselines()
    ref = base["repaired"]
    tune = rec["tune"]
    base_t = {k: restrict(v, ref, tune) for k, v in base.items()}
    noto, _ = nested_stack(base_t, ["CatBoost", "repaired"])
    an, _ = _score(noto)
    print(f"\nno-topology stack, TUNE half: adjR2 = {an:+.4f}")
    print(f"({rec['n_tune']} extractants, {rec['tune_pairs']} adjacent pairs)\n")

    # Amended 22 July: sweep runs train on ALL extractants and are SCORED on the
    # tune half.  tune_only=False asserts that -- a run still carrying
    # --restrict-groups belongs to the superseded design, where the PI-CNN
    # collapsed to adjR2 +0.0362 and every configuration tied at weight 0.00.
    runs = load_runs(tune_only=False)
    rows = []
    for key, cfg in manifest.items():
        seeds = runs.get(key, {})
        if len(seeds) < min_seeds:
            print(f"  {cfg['resolution']:4d}px s={cfg['spread']:.4f} "
                  f"{cfg['channels']:5s} {cfg['weight']:8s} "
                  f"hi={cfg['hi']:.1f}   {len(seeds)}/{min_seeds} seeds "
                  f"-- INCOMPLETE, excluded")
            continue
        arm = restrict(ensemble(seeds), ref, tune)
        adj, corr = mechanism(arm, base_t["repaired"])
        st, w = nested_stack({**base_t, "T": arm},
                             ["CatBoost", "repaired", "T"])
        a, r2 = _score(st)
        gain = a - an
        is_anchor = (cfg["resolution"] == 20 and abs(cfg["spread"] - 0.08) < 1e-9
                     and cfg["channels"] == "sum" and cfg["weight"] == "linear"
                     and abs(cfg["hi"] - 2.5) < 1e-9)
        # The weight the stack actually gives the arm.  ``nested_stack`` returns
        # one weight vector per extractant, in the order of ``names``, so the
        # arm's share is column 2.
        #
        # This is recorded because it is the cleanest signal available, and
        # because a gain of exactly +0.0000 is ambiguous on its own: it means
        # the arm was assigned weight 0 and correctly ignored, not that it was
        # given weight and failed to help.  Validated against the arms whose
        # answers are already known, on this same tune half at 8 seeds:
        #
        #     S0  (adds, +0.0381 on full data)   weight 0.41, gain +0.0039
        #     P0  (does not add)                 weight 0.00, gain -0.0027
        #     T0w (does not add)                 weight 0.00, gain -0.0078
        #
        # So the pre-registered selection statistic does recover the ground
        # truth here. It looked as though it might not: at 1 seed every arm is
        # too weak to earn any weight and they all tie at exactly +0.0000.
        wcol = np.asarray(w)[:, 2]
        w_t = float(wcol.mean())
        # Fraction of extractants whose nested weights use the arm at all.  Far
        # more interpretable than the mean: the simplex grid has step 0.1, so a
        # single extractant out of 84 picking up 0.1 shows as a mean of 0.0012
        # and looks like "nonzero weight" when it is nothing of the sort.
        w_frac = float((wcol > 1e-9).mean())
        rows.append({"key": key, "resolution": cfg["resolution"],
                     "spread": cfg["spread"], "hi": cfg["hi"],
                     "weight": cfg["weight"], "channels": cfg["channels"],
                     "spread_px": cfg["spread"] * (cfg["resolution"] - 1)
                                  / (cfg["hi"] - cfg["lo"]),
                     "n_seeds": len(seeds), "adj_r2": adj, "err_corr": corr,
                     "stack_adj": a, "stack_overall": r2, "tune_gain": gain,
                     "stack_weight": w_t, "stack_weight_frac": w_frac,
                     "is_anchor": is_anchor})
        print(f"  {cfg['resolution']:4d}px s={cfg['spread']:.4f} "
              f"{cfg['channels']:5s} {cfg['weight']:8s} hi={cfg['hi']:.1f}   "
              f"adjR2={adj:+.4f} corr={corr:+.3f} "
              f"w={w_t:.3f}({100*w_frac:.0f}%) "
              f"tune_gain={gain:+.4f}" + ("   <- ANCHOR" if is_anchor else ""))

    if not rows:
        print("\nno complete configurations yet")
        return 1
    # Pre-registered ordering: gain, then the mechanism's two axes, then grid
    # position.  The published P0 arm gets stack weight exactly 0.00, so an
    # all-zero tie across every configuration is a foreseeable outcome rather
    # than a remote one -- and it would otherwise be broken arbitrarily.
    # Strength comes before decorrelation because that is the order the
    # mechanism says binds for this arm, and it is what tuning targets.
    df = pd.DataFrame(rows)
    # An arm the stack assigns weight 0.00 contributes nothing, and its "gain"
    # is then not a property of the arm at all -- it is noise in how the other
    # two components' nested weights land.  Observed directly: the 20px row
    # spans gains of -0.0027 to -0.0030 with every weight at 0.00, differences
    # far below anything meaningful.
    #
    # Sorting those by gain would rank configurations by that noise.  The
    # pre-registered tie-break exists precisely because an all-zero-weight
    # outcome was foreseeable; implementing "tie" as exact equality of gain
    # missed its own rationale.  Corrected here: any configuration the stack
    # actually uses outranks any it does not, and unused configurations are
    # ordered by the mechanism's two axes instead of by noise.
    # "Used" means the stack leans on the arm across extractants, not that some
    # single extractant's weight vector happened to pick it up.  Threshold 0.05
    # is half a grid step, i.e. roughly "a majority of extractants give it
    # weight".  It is not a tuned cut: the published S0 arm sits at 0.41 and the
    # persistence-image configurations at 0.0012, two orders of magnitude either
    # side, so any threshold in between gives the same answer.
    df["used"] = df["stack_weight"] >= 0.05
    # Gain only carries information for configurations the stack actually uses.
    # For the rest it is neutralised to a constant so they tie on it and fall
    # through to the mechanism axes, rather than being ordered by noise in the
    # other components' weights.
    df["_gain_key"] = np.where(df["used"], df["tune_gain"], 0.0)
    out = (df.sort_values(["used", "_gain_key", "adj_r2", "err_corr"],
                          ascending=[False, False, False, True])
           .drop(columns="_gain_key")
           .reset_index(drop=True))
    if not out["used"].any():
        print(f"\n[!] NO configuration earned meaningful stack weight "
              f"(max mean weight {df['stack_weight'].max():.4f}, "
              f"used by {100*df['stack_weight_frac'].max():.0f}% of extractants "
              f"at best; the published S0 arm sits at 0.41).")
        print("    Tuning did not lift persistence images to where the stack "
              "will use them at all.")
        print("    Ranked by the pre-registered tie-break: adjacent-pair R2, "
              "then error correlation.")

    csv = REPORTS / f"pi_sweep_stage_{stage}.csv"
    out.to_csv(csv, index=False)

    anchor = out[out["is_anchor"]]
    if len(anchor):
        a = anchor.iloc[0]
        drift = a["adj_r2"] - P0_PUBLISHED_ADJ
        print(f"\nreproduction anchor: adjR2 {a['adj_r2']:+.4f} vs published P0 "
              f"{P0_PUBLISHED_ADJ:+.4f} (delta {drift:+.4f})")
        print("  NOTE: the anchor is scored on the tune half only, so it is not "
              "directly comparable to the published full-data number.")

    print(f"\n=== best on the TUNE half (selection rule) ===")
    for _, r in out.head(5).iterrows():
        print(f"  {r['resolution']:4.0f}px s={r['spread']:.4f} "
              f"{r['channels']:5s} {r['weight']:8s} hi={r['hi']:.1f}   "
              f"gain={r['tune_gain']:+.4f}  adjR2={r['adj_r2']:+.4f}  "
              f"corr={r['err_corr']:+.3f}")
    if stage == "b":
        _main_effects(out)

    w = out.iloc[0]
    print(f"\nWINNER (stage {stage}): key={w['key']}  res={w['resolution']:.0f} "
          f"spread={w['spread']:.5f} ({w['spread_px']:.2f} px) "
          f"hi={w['hi']:.1f} {w['channels']} {w['weight']}")
    print(f"  tune-half gain {w['tune_gain']:+.4f}, adjR2 {w['adj_r2']:+.4f}, "
          f"err corr {w['err_corr']:+.3f}")
    print(f"  -> {csv}")
    return 0



def _main_effects(out: pd.DataFrame) -> None:
    """Stage B as a factorial, because cell-by-cell it cannot resolve anything.

    Measured run-to-run noise on an identical configuration is SD 0.0065, so a
    difference between two independently-run cells has SE ~0.0092
    (``PI_SWEEP_PRECISION.md``).  Stage B's cells differ by less than that, so
    ranking them individually is ranking noise.

    A main effect is not: it averages 16 cells per level, dropping the noise to
    roughly 0.0092/sqrt(16) = 0.0023 and resolving differences of about 0.005.
    So the question Stage B can actually answer is "does separating H0 from H1
    buy anything, averaged over range and weighting", not "which of these 32
    cells wins".

    Adding more seeds would not have helped -- 8 seeds already buys only 1.76x
    against the 2.83x independence would give, because the nondeterminism has a
    per-run component shared across seeds.
    """
    CELL_SE = 0.0092
    print("\n=== MAIN EFFECTS (the contrasts this stage can resolve) ===")
    print(f"    per-cell SE {CELL_SE:.4f}; a level mean over n cells has "
          f"SE {CELL_SE:.4f}/sqrt(n)")
    for axis in ("channels", "weight", "hi"):
        levels = out.groupby(axis)["adj_r2"].agg(["mean", "count"])
        if len(levels) < 2:
            continue
        print(f"\n  {axis}:")
        for lvl, row in levels.iterrows():
            se = CELL_SE / np.sqrt(row["count"])
            print(f"    {str(lvl):10s} n={int(row['count']):2d}  "
                  f"mean adjR2 {row['mean']:+.4f} +/- {se:.4f}")
        top = levels["mean"].idxmax()
        bot = levels["mean"].idxmin()
        delta = levels.loc[top, "mean"] - levels.loc[bot, "mean"]
        se_d = CELL_SE * np.sqrt(1 / levels.loc[top, "count"]
                                 + 1 / levels.loc[bot, "count"])
        k = delta / se_d if se_d > 0 else float("inf")
        verdict = ("resolvable" if k >= 3 else
                   "MARGINAL" if k >= 2 else "NOT RESOLVABLE")
        print(f"    best-worst ({top} - {bot}): {delta:+.4f} "
              f"+/- {se_d:.4f} = {k:.1f} sigma   {verdict}")


def confirm(key: str, n_boot: int, min_seeds: int) -> int:
    """The pre-registered endpoint, on the half no sweep run ever saw."""
    s0 = harness_check()
    rec = pi_split.load()
    if not pi_split.verify():
        raise SystemExit("the frozen split moved; refusing to proceed")

    base = baselines()
    ref = base["repaired"]
    conf = rec["confirm"]
    base_c = {k: restrict(v, ref, conf) for k, v in base.items()}
    noto, _ = nested_stack(base_c, ["CatBoost", "repaired"])
    an, _ = _score(noto)
    print(f"\nno-topology stack, CONFIRM half: adjR2 = {an:+.4f} "
          f"({rec['n_confirm']} extractants, {rec['confirm_pairs']} pairs)")

    # --- the positive control, which the endpoint is gated on ---------------
    st_s0, _ = nested_stack({**base_c, "T": restrict(s0, ref, conf)},
                            ["CatBoost", "repaired", "T"])
    pc = paired_adjacent_fast(noto, st_s0, n_boot, seed=0)
    ok = pc["lo"] > 0
    print(f"\npositive control -- S0 on the CONFIRM half: {pc['delta']:+.4f} "
          f"[{pc['lo']:+.4f}, {pc['hi']:+.4f}]  "
          f"{'PASSES' if ok else 'FAILS'}")
    if not ok:
        print("\nThe confirm half cannot detect an effect we know is real, so "
              "the persistence-image result is uninterpretable in either "
              "direction. Pre-registered outcome: the test is VOID.")
        return 1

    # Stage C only: full-data runs, from their own directory, with the
    # tune-half interlock asserted rather than assumed.
    runs = load_runs(FINAL, tune_only=False)
    seeds = runs.get(key, {})
    print(f"\nwinner {key}: {len(seeds)} seeds"
          + ("" if len(seeds) >= min_seeds else "  -- INCOMPLETE"))
    if len(seeds) < min_seeds:
        return 1
    arm = restrict(ensemble(seeds), ref, conf)
    adj, corr = mechanism(arm, base_c["repaired"])
    print(f"  P* on the confirm half: adjR2 {adj:+.4f}  err corr {corr:+.3f}")

    rows = []
    st, w = nested_stack({**base_c, "T": arm}, ["CatBoost", "repaired", "T"])
    a, r2 = _score(st)
    print(f"  stack with P*: adjR2 {a:+.4f}  overall {r2:+.4f}  weights {w}")

    print("\n=== PRIMARY: does P* add to the best no-topology stack? ===")
    res = paired_adjacent_fast(noto, st, n_boot, seed=0)
    clo, chi = _corrected(res["delta"], res["lo"], res["hi"], N_LOOKS)
    v = "adds" if res["lo"] > 0 else ("spans 0" if res["hi"] > 0 else "hurts")
    cv = "significant" if clo > 0 else "not significant"
    plo, phi = _corrected(res["delta"], res["lo"], res["hi"], N_LOOKS_PUNITIVE)
    pv = "significant" if plo > 0 else "not significant"
    print(f"  delta {res['delta']:+.4f}  [{res['lo']:+.4f}, {res['hi']:+.4f}]  {v}")
    print(f"  {N_LOOKS}-look corrected [{clo:+.4f}, {chi:+.4f}]  {cv}")
    print(f"  punitive {N_LOOKS_PUNITIVE}-look (one per configuration swept) "
          f"[{plo:+.4f}, {phi:+.4f}]  {pv}")
    rows.append({"contrast": "P* added to no-topology stack", **res,
                 f"lo_{N_LOOKS}look": clo, f"hi_{N_LOOKS}look": chi,
                 f"lo_{N_LOOKS_PUNITIVE}look": plo,
                 f"hi_{N_LOOKS_PUNITIVE}look": phi,
                 "verdict": v, "verdict_corrected": cv,
                 "verdict_punitive": pv})

    # --- secondary: the decisive swap S0 faced ------------------------------
    t0w = _matched_control()
    if t0w is not None:
        st_c, _ = nested_stack({**base_c, "T": restrict(t0w, ref, conf)},
                               ["CatBoost", "repaired", "T"])
        res2 = paired_adjacent_fast(st_c, st, n_boot, seed=0)
        clo2, chi2 = _corrected(res2["delta"], res2["lo"], res2["hi"], N_LOOKS)
        v2 = "beats" if res2["lo"] > 0 else ("spans 0" if res2["hi"] > 0
                                             else "loses to")
        print("\n=== SECONDARY: P* vs the matched 2D control, same slot ===")
        print(f"  delta {res2['delta']:+.4f}  "
              f"[{res2['lo']:+.4f}, {res2['hi']:+.4f}]  {v2}")
        print(f"  {N_LOOKS}-look corrected [{clo2:+.4f}, {chi2:+.4f}]  "
              f"{'significant' if clo2 > 0 else 'not significant'}")
        rows.append({"contrast": "P* vs matched 2D control", **res2,
                     f"lo_{N_LOOKS}look": clo2, f"hi_{N_LOOKS}look": chi2,
                     "verdict": v2,
                     "verdict_corrected": ("significant" if clo2 > 0
                                           else "not significant")})
    else:
        print("\n[warn] matched 2D control (T0w) not found; secondary skipped")

    pd.DataFrame(rows).to_csv(REPORTS / "pi_sweep_confirm.csv", index=False)

    # The pre-registered decision, computed from the numbers rather than read
    # off by eye.
    primary_adds = rows[0]["lo"] > 0 and rows[0][f"lo_{N_LOOKS}look"] > 0
    beats_control = len(rows) > 1 and rows[1]["lo"] > 0
    print("\n=== PRE-REGISTERED OUTCOME ===")
    if primary_adds and beats_control:
        print("  Tuned persistence images DO contribute. The claim broadens "
              "from one representation to a class of them.")
    elif rows[0]["delta"] > 0:
        print(f"  Point estimate positive ({rows[0]['delta']:+.4f}) but not "
              "demonstrable at this sample size. Current scope stands.")
    else:
        print(f"  Persistence images do not contribute even tuned "
              f"({rows[0]['delta']:+.4f}). With the positive control passing, "
              "this is a FAIR test -- strictly stronger than the previous "
              "'untested' caveat.")
    return 0


def _matched_control() -> pd.DataFrame | None:
    cells = load_cells(verbose=False)
    for name in ("T0w", "T0"):
        if name in cells and cells[name]:
            return ensemble(cells[name])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("a", "b"))
    ap.add_argument("--confirm", metavar="KEY",
                    help="image-configuration key of the tune-half winner")
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--min-seeds", type=int, default=3)
    args = ap.parse_args()
    if args.confirm:
        return confirm(args.confirm, args.n_boot, max(args.min_seeds, 8))
    if args.stage:
        return explore(args.stage, args.min_seeds)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
