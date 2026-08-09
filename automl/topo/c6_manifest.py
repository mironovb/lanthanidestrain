"""CAMPAIGN6 cell manifests.

The axes live here, in Python, next to the keys ``lift_report`` groups on --
not in a hand-written JSON.  A 300-cell manifest is not a hand-editable file,
and a manifest whose axis definitions have drifted from the analysis is worse
than no manifest.

    python3 -m automl.topo.c6_manifest --wave w1 --out automl/slurm/manifests
    python3 -m automl.topo.c6_manifest --wave all --dry-run   # just count them

Waves, in the order they should run:

  guard  one cell: re-run a published arm with the new code and prove the
         OOF parquet is byte-identical.  Nothing else launches until it passes.
  w1     the contrast term's shape.  The measured train/eval mismatch, and the
         only item in this campaign that is a defect repair rather than tuning.
  w6     receptive field: radial readout resolution first (it is nearly free
         and the arithmetic says the current 0.258 A bin cannot resolve a
         0.013 A contraction step), then basis range, then the full 4.0 A graph.
  w3     the radius-interaction head.
  w7     arms built to be DECORRELATED rather than individually strong.

Screening protocol: --arch dist (the fastest and the strongest single encoder),
--repeats 1, 4 seeds, restricted to the screen extractants.  Cells are compared
as 4-seed ensembles, never as single runs: the per-seed SD is ~0.047 and an
identical config re-runs 0.0092 apart, so a one-seed lift is not evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "automl/artifacts/c6_split"
# Screening trains and scores on screen+select (106 extractants, 604 pairs).
# The `screen` third alone is 302 pairs, which is thin enough that a 4-seed
# ensemble's noise would swamp the ~0.02 gate.  What has to be protected is the
# REPORT third: no cell is ever ranked on it, so the single final look needs no
# multiplicity correction.  Under leave-extractants-out the final model may
# still TRAIN on those extractants -- what must not happen is choosing on them.
SCREEN = SPLIT / "screen_select_extractants.txt"

# Four screening seeds; the shortlist re-runs on the published 16.
SEEDS = (7, 11, 23, 37)
CONFIRM_SEEDS = (7, 11, 23, 37, 42, 51, 67, 83,
                 211, 223, 233, 241, 251, 263, 271, 281)

# The published contrast objective, unchanged.  Every screening cell carries it
# so a cell measures its own axis and not the objective.
CONTRAST = "--pair-loss-weight 2.0 --select-on adjacent"
# D0: the strongest single arm to date (+0.2474) and the cheapest at ~7.7 min
# for 5x3.  --repeats 1 makes a screening cell ~2.6 min.
BASE = f"--arch dist {CONTRAST} --folds 5 --repeats 1"


def _screen(extra: str) -> str:
    return f"{BASE} {extra} --restrict-groups {SCREEN}".strip()


# --------------------------------------------------------------------------
# w1 -- the shape of the contrast term
#
# --pair-metric-align is the headline: it collapses (block, metal) replicates
# before differencing, exactly as the evaluator does.  The rest of the wave
# exists to say whether alignment alone is enough, and is deliberately small --
# the pair WEIGHT grid runs only after alignment, because alignment changes the
# term's gradient scale and any weight tuned beforehand is re-invalidated.
W1_CELLS = {
    "a0_published":     "",
    "a1_align":         "--pair-metric-align",
    "a2_align_adjonly": "--pair-metric-align --pair-adj-only",
    "a3_align_huber":   "--pair-metric-align --pair-loss-kind huber",
    "a4_align_w10":     "--pair-metric-align --pair-adj-weight 10.0",
    "a5_align_w1":      "--pair-metric-align --pair-adj-weight 1.0",
    # Alignment removes the replicate duplication; does dropping the same-metal
    # pairs matter on its own, without collapsing?  Isolates the two effects.
    "a6_adjonly_only":  "--pair-adj-only",
    "a7_w10_only":      "--pair-adj-weight 10.0",
    "a8_huber_only":    "--pair-loss-kind huber",
    # Alignment + a heavier contrast term.  pair-loss-weight has only ever been
    # 0, 2 and (twice) 5 across 462 recorded runs.
    "a9_align_pw4":     "--pair-metric-align --pair-loss-weight 4.0",
    "a10_align_pw1":    "--pair-metric-align --pair-loss-weight 1.0",
    "a11_align_strict": "--pair-metric-align --block-key strict_composition_key",
}

# --------------------------------------------------------------------------
# w6 -- receptive field, re-opened
#
# The two prior wide-field runs (snn_filt5 -0.0686, snn_allatom -0.3273) both
# predate the contrast objective: their recorded configs carry
# pair_loss_weight=None, select_on=None.  They do not bound anything current.
#
# Priority is resolution before range.  radial_width = radial_max/(bins-1) =
# 8/31 = 0.258 A, against a 0.013 A adjacent contraction step -- a factor of 20.
# The readout's own docstring says it exists to catch sub-0.1 A shell shifts.
W6_CELLS = {
    "b0_rb64":     "--radial-bins 64",
    "b1_rb128":    "--radial-bins 128",
    "b2_rb256":    "--radial-bins 256",
    "b3_rm12":     "--radial-max 12.0",
    "b4_rb128_rm12": "--radial-bins 128 --radial-max 12.0",
    # 24.1% of atoms sit beyond the hardcoded 8.0 A; 18.4% in the 8-10 A shell.
    "b5_rb128_rm10": "--radial-bins 128 --radial-max 10.0",
    # The shipped asset holds edges out to 4.0 A and every run has thresholded
    # them back to 3.5.  Free: no new asset, no new code.
    "b6_f40":      "--filtration-max 4.0",
    "b7_f40_fb64": "--filtration-max 4.0 --rbf-bins 64",
    "b8_fb64":     "--rbf-bins 64",
    "b9_fb64_fm5": "--rbf-bins 64 --rbf-max 5.0",
    # Best-of-resolution crossed with the widest graph the shipped asset allows.
    "b10_f40_rb128": "--filtration-max 4.0 --radial-bins 128",
}

# --------------------------------------------------------------------------
# w3 -- the radius-interaction head (see train.py --radius-slope)
W3_CELLS = {
    "c0_linear":      "--radius-slope linear",
    "c1_quad":        "--radius-slope quad",
    "c2_basis":       "--radius-slope basis",
    "c3_basis_row":   "--radius-slope basis --radius-slope-u row",
    "c4_quad_align":  "--radius-slope quad --pair-metric-align",
    "c5_basis_align": "--radius-slope basis --pair-metric-align",
    # The cell where the identity is exact: no encoder, so u really is
    # ligand+conditions only.  If it does not move here it will not move under
    # an encoder, and that is worth knowing for one cheap run.
    "c6_tabular_basis": "--radius-slope basis",
}

# --------------------------------------------------------------------------
# w4 -- aqueous-phase and f-shell metal constants (preset baseline_2d_mphys)
#
# Pre-screened before any GPU time: automl/reports/c6_prescreen.csv.
# mphys__dG_hyd correlates with dy at 0.215, ABOVE the incumbent
# Ionic Radius_metal (0.171) and above the best of A1's 119 geometry columns
# (0.183); the block's median is 0.139 against A1's 0.0495.  Several columns are
# strongly non-monotone across the series, which is the property the incumbent
# metal columns cannot supply: their within-block adjacent difference is a
# CONSTANT -1, so they carry no information about which pair has the larger dy.
MPHYS = "--preset baseline_2d_mphys"
W4_CELLS = {
    "e0_mphys":            MPHYS,
    "e1_mphys_align":      f"{MPHYS} --pair-metric-align",
    # The interaction head with a richer phi is the natural consumer of this
    # block: g(u) multiplies a metal coordinate, and until now the only clean
    # coordinate available was the radius.
    "e2_mphys_basis":      f"{MPHYS} --radius-slope basis",
    "e3_mphys_basis_align": f"{MPHYS} --radius-slope basis --pair-metric-align",
    "e4_mphys_quad_align": f"{MPHYS} --radius-slope quad --pair-metric-align",
}

# --------------------------------------------------------------------------
# w7 -- arms chosen to be DECORRELATED, not individually strong.
#
# Every prior decorrelation attempt varied the FEATURES and landed at error
# correlation 0.88-0.93, because they all fit the level.  These vary what the
# arm is asked to predict, or strip the fingerprint block the partners already
# carry.
W7_CELLS = {
    "d0_noecfp":     "--preset baseline_no_ecfp",
    "d1_noecfp_align": "--preset baseline_no_ecfp --pair-metric-align",
    "d2_snn_align":  "",          # arch override below
    "d3_g0_align":   "",          # arch override below
}
W7_OVERRIDE = {
    "d2_snn_align": f"--arch snn {CONTRAST} --folds 5 --repeats 1 "
                    f"--pair-metric-align --restrict-groups {SCREEN}",
    "d3_g0_align":  f"--arch snn --no-triangles {CONTRAST} --folds 5 "
                    f"--repeats 1 --pair-metric-align "
                    f"--restrict-groups {SCREEN}",
}


def _cells(mapping: dict[str, str], seeds=SEEDS,
           override: dict[str, str] | None = None) -> list[dict]:
    out = []
    for name, extra in mapping.items():
        for s in seeds:
            if override and name in override:
                args = f"{override[name]} --seed {s}"
            elif name == "c6_tabular_basis":
                args = (f"--arch tabular --match-rows snn {CONTRAST} "
                        f"--folds 5 --repeats 1 {extra} "
                        f"--restrict-groups {SCREEN} --seed {s}")
            else:
                args = f"{_screen(extra)} --seed {s}"
            out.append({"tag": f"c6_{name}_s{s}", "args": args})
    return out


def guard_cells() -> list[dict]:
    """Byte-identity gate: a published arm re-run with the new code.

    d0_dist_s7 in automl/artifacts/topo_encoder is the published DistanceNet
    arm at seed 7.  Re-running it must reproduce its OOF parquet exactly, or
    the "every new flag is default-off" claim is false and nothing downstream
    can be compared to a published number.
    """
    return [{"tag": "d0_dist_s7",
             "args": f"--arch dist {CONTRAST} --folds 5 --repeats 3 --seed 7"}]


def smoke_cells() -> list[dict]:
    """Two epochs of every new code path, before any of them is trusted.

    A crash in --radius-slope discovered 90 cells into a wave costs the wave;
    discovered here it costs three minutes.  One seed, 2 epochs, one repeat --
    these numbers are meaningless and are not read.
    """
    quick = (f"--arch dist {CONTRAST} --folds 5 --repeats 1 --epochs 2 "
             f"--restrict-groups {SCREEN} --seed 7")
    probes = {
        "s_align":     "--pair-metric-align",
        "s_adjonly":   "--pair-metric-align --pair-adj-only",
        "s_huber":     "--pair-loss-kind huber --pair-adj-weight 10.0",
        "s_slope_lin": "--radius-slope linear",
        "s_slope_bas": "--radius-slope basis",
        "s_slope_row": "--radius-slope basis --radius-slope-u row",
        "s_mphys":     "--preset baseline_2d_mphys",
        "s_mphys_bas": "--preset baseline_2d_mphys --radius-slope basis "
                       "--pair-metric-align",
        "s_rbf":       "--rbf-bins 64 --rbf-max 5.0 --radial-bins 128",
        "s_f40":       "--filtration-max 4.0",
    }
    out = [{"tag": f"c6smoke_{k}", "args": f"{quick} {v}"}
           for k, v in probes.items()]
    out.append({"tag": "c6smoke_snn_align",
                "args": f"--arch snn --no-triangles {CONTRAST} --folds 5 "
                        f"--repeats 1 --epochs 2 --restrict-groups {SCREEN} "
                        f"--seed 7 --pair-metric-align --radius-slope quad"})
    out.append({"tag": "c6smoke_tabular",
                "args": f"--arch tabular --match-rows snn {CONTRAST} --folds 5 "
                        f"--repeats 1 --epochs 2 --restrict-groups {SCREEN} "
                        f"--seed 7 --radius-slope basis "
                        f"--preset baseline_2d_mphys"})
    return out


def screen_cells() -> list[dict]:
    """Every screening wave in one manifest, interleaved by seed-block.

    One manifest lets two drivers -- one per GPU partition -- work disjoint
    index ranges of the same list.  Interleaving matters: if the waves were
    concatenated in order, a partition failure would take out one whole axis
    rather than thinning all of them evenly.
    """
    out: list[dict] = []
    for w in ("w1", "w6", "w3", "w4", "w7"):
        out.extend(WAVES[w]())
    # Deterministic round-robin over the axis each cell belongs to.
    buckets: dict[str, list[dict]] = {}
    for c in out:
        buckets.setdefault(c["tag"].split("_")[1][:1], []).append(c)
    order, i = [], 0
    while any(buckets.values()):
        for k in sorted(buckets):
            if buckets[k]:
                order.append(buckets[k].pop(0))
        i += 1
    return order


WAVES = {
    "smoke": smoke_cells,
    "guard": guard_cells,
    "w1": lambda: _cells(W1_CELLS),
    "w6": lambda: _cells(W6_CELLS),
    "w3": lambda: _cells(W3_CELLS),
    "w4": lambda: _cells(W4_CELLS),
    "w8": lambda: _cells(W8_CELLS),
    "w9": lambda: _cells(W9_CELLS),
    "w7": lambda: _cells(W7_CELLS, override=W7_OVERRIDE),
}
WAVES["screen"] = screen_cells


# --------------------------------------------------------------------------
# z -- compositions of the screening winners.
#
# NOT screened themselves; each combines axes that independently cleared the
# +0.02 gate, and they are labelled as compositions wherever they are reported.
# The axes are mechanically independent (a feature block, a graph cutoff, a
# loss weight), so composing them is a reasonable bet -- but it is still a bet,
# and the confirm stage is where it is paid for.
#
# Deliberately EXCLUDED: --pair-metric-align. All 16 aligned cells scored at or
# below the unaligned control, so the composition inherits the reweighting that
# worked (--pair-adj-weight 10) without the replicate collapse that did not.
COMBO = {
    "z0_mphys_f40": f"{MPHYS} --filtration-max 4.0 --rbf-bins 64",
    "z1_mphys_f40_w10": f"{MPHYS} --filtration-max 4.0 --rbf-bins 64 "
                        f"--pair-adj-weight 10.0",
    "z2_mphys_w10": f"{MPHYS} --pair-adj-weight 10.0",
    "z3_mphys_f40_rb128": f"{MPHYS} --filtration-max 4.0 --rbf-bins 64 "
                          f"--radial-bins 128",
}


# --------------------------------------------------------------------------
# w8 -- past the shipped 4.0 A ceiling, and past emphasis 10.
#
# Motivated by the confirmed screening result, not by speculation: 4.0 A beat
# 3.5 A and emphasis 10 beat emphasis 3, and BOTH were at the edge of what the
# existing asset and the existing grid allowed.  These cells ask whether the
# two axes were still climbing when they ran out of room.
#
# All cells carry the endpoint's feature block (mphys) so the axis is measured
# on top of the best-known arm rather than against a weaker one -- the
# "test against the champion, not the convenience baseline" rule that this
# study learned the hard way four times.
W8_CELLS = {
    # wider graphs.  --filtration-max must move with the asset or the load path
    # thresholds the new edges straight back off.
    "y0_c50":  f"{MPHYS} --edge-asset c50 --filtration-max 5.0 --rbf-bins 64 --rbf-max 5.0",
    "y1_c60":  f"{MPHYS} --edge-asset c60 --filtration-max 6.0 --rbf-bins 96 --rbf-max 6.0",
    "y2_c80":  f"{MPHYS} --edge-asset c80 --filtration-max 8.0 --rbf-bins 128 --rbf-max 8.0",
    # degree-based rather than distance-based neighbourhood
    "y3_k24":  f"{MPHYS} --edge-asset k24 --filtration-max 12.0 --rbf-bins 96 --rbf-max 8.0",
    # emphasis past 10
    "y4_w20":  f"{MPHYS} --filtration-max 4.0 --rbf-bins 64 --pair-adj-weight 20.0",
    "y5_w40":  f"{MPHYS} --filtration-max 4.0 --rbf-bins 64 --pair-adj-weight 40.0",
    # wider graph AND higher emphasis
    "y6_c50_w20": f"{MPHYS} --edge-asset c50 --filtration-max 5.0 --rbf-bins 64 "
                  f"--rbf-max 5.0 --pair-adj-weight 20.0",
    "y7_c60_w20": f"{MPHYS} --edge-asset c60 --filtration-max 6.0 --rbf-bins 96 "
                  f"--rbf-max 6.0 --pair-adj-weight 20.0",
}


# --------------------------------------------------------------------------
# w9 -- robustness of the LEVEL term, the neural analogue of the one thing that
# worked.
#
# Swapping CatBoost's RMSE for MAE was worth +0.1066 adjacent and +0.0115 log D,
# and survived on the held-out third at +0.0552 [+0.0177, +0.1009].  The neural
# arms have used Huber(delta=1.0) on the standardised target in all 462 recorded
# runs.  The metric is a within-block DIFFERENCE, so one badly-measured row
# corrupts every pair it enters -- bounding each row's influence is exactly the
# lever MAE pulled for the tree.
#
# --pair-loss-kind huber (the PAIR term) did nothing, -0.0161, which localises
# the leverage to the level fit and is why this wave moves the level term only.
#
# Run against the plain published arm AND against the mphys block, so a gain
# here is not confounded with the feature block that failed to generalise.
W9_CELLS = {
    "x0_mae":        "--level-loss mae",
    "x1_d05":        "--level-huber-delta 0.5",
    "x2_d02":        "--level-huber-delta 0.2",
    "x3_mse":        "--level-loss mse",
    "x4_mae_mphys":  f"{MPHYS} --level-loss mae",
    "x5_d02_mphys":  f"{MPHYS} --level-huber-delta 0.2",
    "x6_mae_f40":    "--level-loss mae --filtration-max 4.0 --rbf-bins 64",
    "x7_d02_w10":    "--level-huber-delta 0.2 --pair-adj-weight 10.0",
}


def confirm_cells(shortlist: list[str]) -> list[dict]:
    """The shortlist at 16 seeds, 5x3 folds, on the FULL 162 extractants.

    No --restrict-groups here.  Under leave-extractants-out every extractant is
    held out in some fold, so training on all of them still gives honest
    out-of-fold predictions for the report third -- what had to be protected is
    the CHOOSING, and that was done on screen+select only.  Scoring is then
    restricted at analysis time by c6_final --report.

    ``shortlist`` entries are the cell names from the screening manifest, e.g.
    "a1_align".  Their argument strings are looked up so the confirmed cell is
    the screened cell rather than a hand-retyped approximation of it.
    """
    lookup: dict[str, str] = {}
    for mapping, override in ((W1_CELLS, None), (W6_CELLS, None),
                              (W3_CELLS, None), (W4_CELLS, None),
                              (COMBO, None), (W8_CELLS, None),
                              (W9_CELLS, None),
                              (W7_CELLS, W7_OVERRIDE)):
        for name, extra in mapping.items():
            if override and name in override:
                base = override[name]
            elif name == "c6_tabular_basis":
                base = (f"--arch tabular --match-rows snn {CONTRAST} {extra}")
            else:
                base = f"--arch dist {CONTRAST} {extra}".strip()
            # Strip the screening-only settings; confirmation is full protocol.
            base = base.replace(f"--restrict-groups {SCREEN}", "")
            base = base.replace("--folds 5 --repeats 1", "").strip()
            lookup[name] = " ".join(base.split())
    missing = [s for s in shortlist if s not in lookup]
    if missing:
        raise SystemExit(f"unknown cell name(s) {missing}; "
                         f"have {sorted(lookup)}")
    out = []
    for name in shortlist:
        for s in CONFIRM_SEEDS:
            out.append({"tag": f"c6c_{name}_s{s}",
                        "args": f"{lookup[name]} --folds 5 --repeats 3 "
                                f"--seed {s}"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", nargs="+", default=None,
                    help="cell names to confirm at 16 seeds on the full data")
    ap.add_argument("--wave", default="all",
                    help="one of " + ", ".join(WAVES) + ", or 'all'")
    ap.add_argument("--out", default=str(REPO / "automl/slurm/manifests"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SCREEN.exists():
        raise SystemExit(f"missing {SCREEN}; run "
                         f"python3 -m automl.topo.c6_split first")

    if args.confirm:
        cells = confirm_cells(list(args.confirm))
        out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
        pth = out_dir / "c6_confirm.json"
        if not args.dry_run:
            pth.write_text(json.dumps(cells, indent=1) + "\n")
        print(f"[c6] confirm: {len(args.confirm)} cells x "
              f"{len(CONFIRM_SEEDS)} seeds = {len(cells)} runs -> {pth}")
        for c in cells[:2]:
            print(f"       e.g. {c['tag']}: {c['args']}")
        return 0

    waves = list(WAVES) if args.wave == "all" else [args.wave]
    bad = [w for w in waves if w not in WAVES]
    if bad:
        raise SystemExit(f"unknown wave(s) {bad}; have {list(WAVES)}")

    out_dir = Path(args.out)
    total = 0
    for w in waves:
        cells = WAVES[w]()
        tags = [c["tag"] for c in cells]
        if len(set(tags)) != len(tags):
            raise SystemExit(f"wave {w} has duplicate tags; the .done sentinel "
                             f"is keyed on the tag and would collide")
        total += len(cells)
        print(f"[c6] wave {w}: {len(cells)} cells "
              f"({len(cells) // max(len(SEEDS), 1)} configs x seeds)")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / f"c6_{w}.json"
            p.write_text(json.dumps(cells, indent=1) + "\n")
            print(f"       -> {p}")
    print(f"[c6] {total} cells total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
