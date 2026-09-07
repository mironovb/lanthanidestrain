#!/usr/bin/env python3
"""Three-Hamiltonian contraction benchmark: GFN2-xTB, g-xTB, Ln-xTB.

Per-ligand compliance c_L = d<M-donor>/d r_Shannon (1.00 = Shannon radii),
fitted exactly as automl/qc/compliance_test.py does, on the same 71 anchors
x 15 metals.  Ln-xTB arms: lnxtb_hs (the SI's spin protocol) and lnxtb_cs
(closed-shell control).  Reports mean +- sd per arm, one-sample t against
1.00, paired contrasts between arms, and the count of ligands on which each
arm is closer to 1.00 than GFN2.  Also the fraction of ligands where the
optimisation succeeded, and the median linear-fit R^2 (how straight the
series is under each Hamiltonian).

    python3 -m automl.qc.lnxtb_summary --tags cf_shard0,cf_shard1,lnxtb_shard0,lnxtb_shard1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from automl.qc.compliance_test import SERIES, compliance, load_series

OUT = SERIES / "lnxtb_benchmark.json"
NAMES = {"gfn2": "GFN2-xTB", "gxtb_hs": "g-xTB", "lnxtb_hs": "Ln-xTB",
         "lnxtb_cs": "Ln-xTB closed-shell"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", default="cf_shard0,cf_shard1,lnxtb_shard0,lnxtb_shard1")
    args = ap.parse_args()
    recs = load_series([t for t in args.tags.split(",") if t])
    R = pd.DataFrame(recs)
    ok_rate = R.groupby("arm")["ok"].mean().round(4).to_dict()
    C = compliance([r for r in recs if r.get("ok")])
    cw = C.pivot(index="family", columns="arm", values="c_L")
    fr = C.pivot(index="family", columns="arm", values="fit_r2")
    arms = [a for a in NAMES if a in cw.columns]
    common = cw[arms].dropna()
    out = {"arms": arms, "n_ligands_common": int(len(common)),
           "n_ligands_per_arm": {a: int(cw[a].notna().sum()) for a in arms},
           "opt_success_rate": ok_rate, "per_arm": {}, "paired": {}}
    print(f"{len(common)} ligands with all arms fitted; per-arm n = {out['n_ligands_per_arm']}")
    print(f"{'arm':22s} {'c_L mean':>9s} {'sd':>7s} {'t vs 1.00':>10s} {'p':>9s} {'median fit R2':>14s} {'ok rate':>8s}")
    for a in arms:
        v = common[a].to_numpy()
        t, p = stats.ttest_1samp(v, 1.0)
        out["per_arm"][a] = {"name": NAMES[a], "mean": round(float(v.mean()), 4),
                             "sd": round(float(v.std(ddof=1)), 4),
                             "t_vs_1": round(float(t), 2), "p_vs_1": float(p),
                             "median_fit_r2": round(float(fr.loc[common.index, a].median()), 4)}
        print(f"{NAMES[a]:22s} {v.mean():9.3f} {v.std(ddof=1):7.3f} {t:+10.1f} {p:9.1e} "
              f"{fr.loc[common.index, a].median():14.3f} {ok_rate.get(a, float('nan')):8.3f}")
    for a in arms:
        for b in arms:
            if a >= b or a == b:
                continue
            da = np.abs(common[a] - 1.0); db = np.abs(common[b] - 1.0)
            t, p = stats.ttest_rel(common[b], common[a])
            out["paired"][f"{b}_vs_{a}"] = {
                "mean_diff": round(float((common[b] - common[a]).mean()), 4),
                "t": round(float(t), 2), "p": float(p),
                "b_closer_to_1_count": int((db < da).sum()), "n": int(len(common))}
            print(f"  {NAMES[b]} vs {NAMES[a]}: mean diff {float((common[b]-common[a]).mean()):+.3f}, "
                  f"paired t {t:+.1f} (p {p:.1e}); {NAMES[b]} closer to 1.00 on {(db < da).sum()}/{len(common)}")
    # Gd break: |deviation of Gd from the Eu/Tb midpoint| in the mean donor distance, per ligand
    okr = R[R.ok == True]
    gd = {}
    for a in arms:
        vals = []
        for fam, g in okr[okr.arm == a].groupby("family"):
            m = g.set_index("metal")["mean_m_donor"]
            if all(k in m.index for k in ("Eu", "Gd", "Tb")):
                vals.append(m["Gd"] - 0.5 * (m["Eu"] + m["Tb"]))
        if vals:
            gd[a] = {"n": len(vals), "mean_A": round(float(np.mean(vals)), 4),
                     "sd_A": round(float(np.std(vals, ddof=1)), 4)}
    out["gd_break_A"] = gd
    print("Gd deviation from the Eu/Tb midpoint (Å):", json.dumps(gd))
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
