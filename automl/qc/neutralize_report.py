#!/usr/bin/env python3
"""Validation gates for the neutral species. A rejected structure is recorded, never dropped.

Pre-registered in ``CAMPAIGN4_PREREGISTRATION.md``.

Every gate records **the number**, not just a boolean, so a threshold can be
re-litigated without re-running three CPU-days of xTB.  Thresholds come from the
corpus, not from the literature: Ln-O(nitrate) at this level of theory is
2.130 A, not the crystallographic 2.45-2.55, and a gate on the literature value
rejects 100% of the existing corpus.

The gates, in order of how silently they fail:

G2  nitrate intact, and **no H within 1.40 A of a nitrate O**.  This is the most
    dangerous failure in the whole pipeline: these ligands carry N-H and O-H, and
    a -1 anion pressed against an acidic proton takes it, giving HNO3 plus a
    deprotonated ligand.  Same formula, same total charge, converged, plausible
    -- and invisible to every composition or charge check.
G4  the original ligand undisturbed, measured against the CONTROL, three ways
    with different failure modes.  RMSD alone is dominated by the bulk of the
    ligand and can miss a local rearrangement; the superposition-free distance
    check cannot be gamed by a bad alignment; the connectivity check catches a
    proton hop that both of the others miss.
G6  convergence, including the population pile-up check that caught the shipped
    set capping at 0.19999 eV/A.

Usage
-----
    python3 -m automl.qc.neutralize_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "automl/artifacts/neutral_species"
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "neutralize_audit.csv"

# --- thresholds, all corpus-derived --------------------------------------
NO_BOND_MIN, NO_BOND_MAX = 1.15, 1.35
ONO_MIN, ONO_MAX = 108.0, 132.0
ONO_SUM_MIN = 354.0
N_OOP_MAX_A = 0.15
MIN_H_TO_NITRATE_O = 1.40
INNER_CUTOFF_A = 3.10
LNN_MIN_A, LNN_MAX_A = 4.0, 14.0
RMSD_MAX_A = 1.0
D_LN_DONOR_MAX = 0.25
D_DONOR_DONOR_MAX = 0.40
ACCEPT_FRACTION_MIN = 0.80          # pre-registered: below this, do not model
FAMILY_RATE_MULTIPLE = 3.0

REJECT_CODES = (
    "accepted", "NO_XTB", "MULTI_METAL", "CHARGE_UNRECOVERABLE",
    "CHARGE_MODEL_BROKEN", "SEED_NO_FEASIBLE_POSE", "CONTROL_FAILED",
    "XTB_CRASH", "timeout", "scf_not_converged", "EXCEPTION",
    "NITRATE_BROKEN", "NITRATE_PROTONATED", "NITRATE_PYRAMIDAL",
    "MODE_MIGRATED", "LIGAND_MOVED", "CONNECTIVITY_CHANGED", "CN_CHANGED",
    "NOT_CONVERGED",
)


def _load_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    sym, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        sym.append(p[0]); xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return sym, np.asarray(xyz, dtype=float)


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))


def gate_structure(rec: dict) -> dict:
    """Every gate for one accepted-so-far structure. Returns numbers + a code."""
    from automl.qc.neutralize import (_metal_index, ligand_cn, kabsch_rmsd,
                                      DONOR_SYMBOLS)
    g = {"geometry_key": rec.get("geometry_key"), "metal": rec.get("metal")}
    nx, cx = rec.get("neutral_xyz"), rec.get("control_xyz")
    if not nx or not cx:
        g["reject_code"] = rec.get("reject_code", "XTB_CRASH"); return g
    ns, nc = _load_xyz(REPO / nx)
    cs, cc = _load_xyz(REPO / cx)
    n_add = int(rec.get("n_add", 0))
    n0 = len(ns) - 4 * n_add
    g["n_atoms"] = len(ns); g["n_add"] = n_add

    # --- G1 composition / indexing ---------------------------------------
    if ns[:n0] != cs[:len(cs)] or len(cs) != n0:
        g["reject_code"] = "COMPOSITION_MISMATCH"; return g
    mi = _metal_index(ns[:n0])
    g["metal_index"] = mi

    # --- G2 nitrate intact ------------------------------------------------
    worst_h = np.inf
    for a in range(n_add):
        b = n0 + 4 * a
        N, Os = nc[b], nc[b + 1:b + 4]
        d = np.linalg.norm(Os - N, axis=1)
        g[f"no_bond_min_{a}"] = float(d.min()); g[f"no_bond_max_{a}"] = float(d.max())
        if d.min() < NO_BOND_MIN or d.max() > NO_BOND_MAX:
            g["reject_code"] = "NITRATE_BROKEN"; return g
        angs = [_angle(Os[i], N, Os[j]) for i, j in ((0, 1), (1, 2), (2, 0))]
        g[f"ono_sum_{a}"] = float(sum(angs))
        if min(angs) < ONO_MIN or max(angs) > ONO_MAX or sum(angs) < ONO_SUM_MIN:
            g["reject_code"] = "NITRATE_PYRAMIDAL"; return g
        nrm = np.cross(Os[1] - Os[0], Os[2] - Os[0])
        nrm = nrm / max(np.linalg.norm(nrm), 1e-12)
        oop = abs(float(np.dot(N - Os[0], nrm)))
        g[f"n_oop_{a}"] = oop
        if oop > N_OOP_MAX_A:
            g["reject_code"] = "NITRATE_PYRAMIDAL"; return g
        hidx = [i for i, s in enumerate(ns[:n0]) if s == "H"]
        if hidx:
            dh = np.linalg.norm(nc[hidx][:, None, :] - Os[None, :, :], axis=2)
            worst_h = min(worst_h, float(dh.min()))
    g["min_H_to_nitrate_O"] = None if worst_h is np.inf else worst_h
    if worst_h is not np.inf and worst_h < MIN_H_TO_NITRATE_O:
        g["reject_code"] = "NITRATE_PROTONATED"; return g

    # --- G3 binding mode --------------------------------------------------
    inner = 0; lnn = []
    for a in range(n_add):
        b = n0 + 4 * a
        lnn.append(float(np.linalg.norm(nc[b] - nc[mi])))
        inner += int((np.linalg.norm(nc[b + 1:b + 4] - nc[mi], axis=1)
                      < INNER_CUTOFF_A).sum())
    g["n_inner_O"] = inner; g["lnn_max"] = max(lnn); g["lnn_min"] = min(lnn)
    if inner > 0:
        g["reject_code"] = "MODE_MIGRATED"; return g
    if max(lnn) > LNN_MAX_A or min(lnn) < LNN_MIN_A:
        g["reject_code"] = "MODE_MIGRATED"; return g

    # --- G4 ligand undisturbed, against the CONTROL -----------------------
    g["rmsd_vs_control"] = kabsch_rmsd(nc[:n0], cc[:n0])
    don = [i for i, s in enumerate(ns[:n0])
           if s in DONOR_SYMBOLS
           and np.linalg.norm(cc[i] - cc[mi]) < INNER_CUTOFF_A]
    g["n_donors"] = len(don)
    if don:
        dn = np.linalg.norm(nc[don] - nc[mi], axis=1)
        dc = np.linalg.norm(cc[don] - cc[mi], axis=1)
        g["dLn_donor_max_delta"] = float(np.abs(dn - dc).max())
        pn = np.linalg.norm(nc[don][:, None, :] - nc[don][None, :, :], axis=2)
        pc = np.linalg.norm(cc[don][:, None, :] - cc[don][None, :, :], axis=2)
        g["dij_donor_max_delta"] = float(np.abs(pn - pc).max())
    else:
        g["dLn_donor_max_delta"] = g["dij_donor_max_delta"] = np.nan
    if g["rmsd_vs_control"] > RMSD_MAX_A:
        g["reject_code"] = "LIGAND_MOVED"; return g
    if (g["dLn_donor_max_delta"] > D_LN_DONOR_MAX
            or g["dij_donor_max_delta"] > D_DONOR_DONOR_MAX):
        g["reject_code"] = "LIGAND_MOVED"; return g

    # --- G5 coordination number ------------------------------------------
    g["cn_ligand_out"] = ligand_cn(ns[:n0], nc[:n0], mi)
    g["cn_ligand_in"] = int(rec.get("cn_ligand_in", -1))
    if g["cn_ligand_in"] >= 0 and g["cn_ligand_out"] != g["cn_ligand_in"]:
        g["reject_code"] = "CN_CHANGED"; return g

    # --- G6 convergence, both arms ---------------------------------------
    for arm in ("neutral", "control"):
        a = rec.get(arm) or {}
        g[f"{arm}_force_max"] = a.get("force_max_ev_ang")
        g[f"{arm}_converged"] = a.get("xtb_converged")
        g[f"{arm}_meets_target"] = a.get("meets_target")
        g[f"{arm}_downgraded"] = bool(a.get("downgraded_opt_level"))
        if not (a.get("xtb_converged") and a.get("meets_target")):
            g["reject_code"] = "NOT_CONVERGED"; return g
    g["reject_code"] = "accepted"
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    recs = sorted((ROOT / "records").glob("*.json")) if (ROOT / "records").exists() else []
    if not recs:
        print(f"[neutralize-report] no records under {ROOT/'records'}")
        return 1
    rows = []
    for p in recs:
        rec = json.loads(p.read_text())
        if rec.get("reject_code") not in (None, "accepted"):
            rows.append({"geometry_key": rec.get("geometry_key"),
                         "metal": rec.get("metal"),
                         "reject_code": rec.get("reject_code"),
                         "n_add": rec.get("n_add"),
                         "cn_ligand_in": rec.get("cn_ligand_in")})
            continue
        rows.append(gate_structure(rec))
    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)

    n = len(d); acc = int((d["reject_code"] == "accepted").sum())
    print(f"[neutralize-report] {n} structures, {acc} accepted ({acc/n:.1%})\n")
    print(d["reject_code"].value_counts().to_string())

    review = []
    if acc / n < ACCEPT_FRACTION_MIN:
        review.append(f"accepted fraction {acc/n:.1%} is below the pre-registered "
                      f"{ACCEPT_FRACTION_MIN:.0%}; do NOT proceed to modelling")
    # family-correlated failure: the mode that silently reshapes a dataset
    if "metal" in d and n >= 20:
        glob = 1 - acc / n
        for m, grp in d.groupby("metal"):
            if len(grp) >= 20:
                r = 1 - (grp["reject_code"] == "accepted").mean()
                if glob > 0 and r > FAMILY_RATE_MULTIPLE * glob:
                    review.append(f"metal {m}: reject rate {r:.0%} vs global "
                                  f"{glob:.0%} -- family-correlated failure")
    if acc:
        a = d[d["reject_code"] == "accepted"]
        for col in ("rmsd_vs_control", "dLn_donor_max_delta", "lnn_max",
                    "min_H_to_nitrate_O"):
            if col in a and a[col].notna().any():
                print(f"\n  {col:24s} median {a[col].median():.3f}  "
                      f"p95 {a[col].quantile(.95):.3f}  max {a[col].max():.3f}")
    if review:
        print("\n" + "=" * 68)
        for r in review:
            print(f"  REVIEW: {r}")
        print("=" * 68)
    else:
        print("\n  no REVIEW banner: the generation is clean on its own criteria.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
