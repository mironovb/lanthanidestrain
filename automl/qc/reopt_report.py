#!/usr/bin/env python3
"""Stage 2 audit: did tighter, solvated re-optimisation change the geometries,
and did it change them *legitimately*?

Two questions, and they pull in opposite directions.

1. **Did convergence actually improve?**  The shipped structures pile up against
   an ``fmax = 0.2 eV/A`` ceiling -- they are not relaxed structures, they are
   structures that ran out of criterion.  If the re-optimised set still piles up
   anywhere, the new criterion is binding too and nothing was gained.

2. **Did the chemistry survive?**  The pilot showed RMSD up to 3.2 A, which is a
   large move.  A re-optimisation that quietly changes coordination number,
   ejects a donor, or dissociates a ligand has not improved the geometry -- it
   has replaced the molecule, and every descriptor downstream would be
   describing something the dataset never measured.  ``AGENTS.md`` is explicit
   that CN, donor set and ligand identity are carried through, never re-decided,
   so this is a hard audit and not a diagnostic nicety.

Nothing here writes to ``data/``; it reads the re-optimised artifacts and the
original geometries and emits a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.geometry_features import (                   # noqa: E402
    read_extxyz, ATOMIC_NUMBER, LANTHANIDE_SYMBOLS, DONOR_SYMBOLS)
from automl.geom3d_features import DONOR_CUTOFF_A      # noqa: E402
from automl.qc.reoptimize import OUT_ROOT, job_table   # noqa: E402

SHIPPED_FMAX_CEILING = 0.2


def _symbols_to_z(symbols) -> np.ndarray:
    return np.array([ATOMIC_NUMBER.get(str(s), 0) for s in symbols], dtype=int)


def coordination(symbols, coords, cutoff: float = DONOR_CUTOFF_A):
    """Donor set around the metal, using the repo's existing 3.10 A convention.

    Returns (metal_index, sorted donor element list, CN).  Using the same cutoff
    as ``geom3d_features`` means a CN change reported here is a CN change the
    descriptor pipeline would also see -- not an artefact of a different rule.
    """
    # Metal and donor identification reuse the repo's own symbol sets, so a CN
    # reported here is the CN the descriptor pipeline would compute.
    metal_pos = [i for i, s in enumerate(symbols)
                 if str(s) in LANTHANIDE_SYMBOLS]
    if len(metal_pos) != 1:
        return None, [], -1
    mi = metal_pos[0]
    d = np.linalg.norm(np.asarray(coords) - np.asarray(coords)[mi], axis=1)
    donors = [i for i in range(len(symbols))
              if i != mi and d[i] <= cutoff and str(symbols[i]) in DONOR_SYMBOLS]
    return mi, sorted(str(symbols[i]) for i in donors), len(donors)


def audit(solvent: str) -> pd.DataFrame:
    jobs = job_table().set_index("basename")
    rows = []
    for js in sorted((OUT_ROOT / solvent).glob("*.json")):
        rec = json.loads(js.read_text())
        base = rec.get("basename")
        row = {k: rec.get(k) for k in
               ("basename", "metal", "charge", "n_atoms", "ok", "reason",
                "seconds", "cycles", "xtb_converged", "meets_target",
                "force_max_ev_ang", "input_force_max_ev_ang",
                "rmsd_from_input_ang", "energy_ev")}
        row["solvent"] = solvent
        if rec.get("ok") and base in jobs.index:
            try:
                g_old = read_extxyz(Path(jobs.loc[base, "local"]))
                g_new = read_extxyz(Path(rec["xyz"]))
                _, d_old, cn_old = coordination(g_old.symbols, g_old.coordinates)
                _, d_new, cn_new = coordination(g_new.symbols, g_new.coordinates)
                row.update(cn_before=cn_old, cn_after=cn_new,
                           cn_changed=(cn_old != cn_new),
                           donors_before="".join(d_old),
                           donors_after="".join(d_new),
                           donor_set_changed=(d_old != d_new),
                           formula_preserved=(sorted(g_old.symbols)
                                              == sorted(g_new.symbols)))
            except Exception as exc:                    # audit must not hide
                row["audit_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    out = []
    A = out.append
    n = len(df)
    ok = df["ok"].fillna(False).astype(bool)
    A(f"structures with a status file : {n}")
    A(f"  succeeded                   : {int(ok.sum())}")
    A(f"  failed                      : {int((~ok).sum())}")
    if (~ok).any():
        A("  failure reasons:")
        for r, c in df.loc[~ok, "reason"].value_counts().items():
            A(f"    {c:5d}  {r}")
    d = df[ok]
    if d.empty:
        return "\n".join(out)

    A("")
    A("-- convergence --------------------------------------------------------")
    f_in = pd.to_numeric(d["input_force_max_ev_ang"], errors="coerce")
    f_out = pd.to_numeric(d["force_max_ev_ang"], errors="coerce")
    A(f"  input  fmax  median {f_in.median():.4f}  "
      f"frac within 1% of the {SHIPPED_FMAX_CEILING} ceiling: "
      f"{float((f_in > 0.99 * SHIPPED_FMAX_CEILING).mean()):.3f}")
    A(f"  output fmax  median {f_out.median():.5f}  max {f_out.max():.5f}")
    A(f"  meets target                : {int(d['meets_target'].sum())}/{len(d)}")
    A(f"  xtb reported converged      : {int(d['xtb_converged'].sum())}/{len(d)}")
    # The point of the check: no residual pile-up at any ceiling.
    top = float(f_out.max())
    A(f"  frac within 1% of own max   : {float((f_out > 0.99 * top).mean()):.4f}"
      f"   (a pile-up here means the new criterion is binding too)")

    A("")
    A("-- geometry change ----------------------------------------------------")
    r = pd.to_numeric(d["rmsd_from_input_ang"], errors="coerce")
    A(f"  RMSD from input (A): median {r.median():.3f}  p90 {r.quantile(0.9):.3f}"
      f"  max {r.max():.3f}")
    if "metal" in d:
        g = d.groupby("metal")["rmsd_from_input_ang"].median().sort_values()
        A(f"  by metal, median RMSD: lowest {g.index[0]}={g.iloc[0]:.3f}, "
          f"highest {g.index[-1]}={g.iloc[-1]:.3f}")

    A("")
    A("-- chemistry preserved (the hard audit) -------------------------------")
    if "audit_error" in d and d["audit_error"].notna().any():
        n_err = int(d["audit_error"].notna().sum())
        A(f"  !! AUDIT DID NOT RUN on {n_err}/{len(d)} structures:")
        for e, c in d["audit_error"].dropna().value_counts().head(3).items():
            A(f"     {c:5d}  {e}")
        A("     An audit that silently does not run looks identical to an")
        A("     audit that passed.  Treat this as a failure, not a gap.")
    if "cn_changed" not in d:
        A("  !! NO CN COMPARISON PRODUCED -- audit incomplete, do not proceed.")
    if "cn_changed" in d:
        cc = d["cn_changed"].fillna(False)
        ds = d["donor_set_changed"].fillna(False)
        fp = d["formula_preserved"].fillna(True)
        A(f"  CN changed                  : {int(cc.sum())}/{len(d)} "
          f"({100*float(cc.mean()):.1f}%)")
        A(f"  donor element set changed   : {int(ds.sum())}/{len(d)} "
          f"({100*float(ds.mean()):.1f}%)")
        A(f"  formula NOT preserved       : {int((~fp).sum())}  "
          f"(must be 0 -- atoms are never added or removed)")
        if cc.any():
            A("  CN transitions (before -> after), most common first:")
            t = (d.loc[cc, ["cn_before", "cn_after"]]
                 .astype("Int64").astype(str).agg(" -> ".join, axis=1)
                 .value_counts().head(8))
            for k, v in t.items():
                A(f"    {v:5d}  CN {k}")
            A("  NOTE: a CN change means the re-optimised structure is not the")
            A("        complex the dataset row describes.  These rows must be")
            A("        excluded or reported, never silently featurised.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvents", default="water,octanol")
    ap.add_argument("--out-dir", default=str(_REPO / "automl/artifacts/geom_reopt"))
    args = ap.parse_args()

    frames = []
    for s in [x for x in args.solvents.split(",") if x]:
        if not (OUT_ROOT / s).exists():
            print(f"[reopt-report] no artifacts for solvent {s!r} yet")
            continue
        df = audit(s)
        if df.empty:
            print(f"[reopt-report] {s}: no status files yet")
            continue
        print(f"\n{'='*72}\n{s.upper()}\n{'='*72}")
        print(report(df))
        frames.append(df)

    if frames:
        allf = pd.concat(frames, ignore_index=True)
        out = Path(args.out_dir) / "reopt_audit.parquet"
        allf.to_parquet(out, index=False)
        print(f"\n[reopt-report] wrote {out}  ({len(allf)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
