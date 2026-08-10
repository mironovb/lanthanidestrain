#!/usr/bin/env python3
"""Does a better electronic-structure method see f-shell structure that GFN2 cannot?

Why this exists
---------------
C-I is settled: inside GFN2-xTB every lanthanide parameter from Ce(58) to Lu(71)
is *linear interpolation between two fitted anchors* (max residual 5e-7, the
file's printed precision).  The authors say so themselves.  Consequence: the
identity of the lanthanide is **one scalar, linear in Z**.  No f-shell, no
crystal field, no gadolinium break, no tetrad effect.  Any geometry GFN2
produces can carry at most a rank-1 linear-in-Z deformation, which is exactly
the effective rank 1.05-of-8 measured across eight independent 3D encoders.

That makes the ceiling a property of the *method*, not of our architectures --
and therefore possibly removable.  g-xTB (Grimme, 2025) is the candidate: it
fits f-shell Mulliken populations of the lanthanides as targets and carries
f-projector ACPs, i.e. the f electrons are in the valence rather than the core.

The observable, and why the obvious one is wrong
------------------------------------------------
My first attempt used the **total energy** and found a huge La->Lu span
(1204 Eh vs GFN2's 0.153 Eh).  That number is real but useless: it is dominated
by the isolated-atom f-electron energy, which says nothing about bonding.  Any
"non-linearity" in it is non-linearity of the free ion.

The fix is an interaction energy that cancels the atomic reference exactly:

    E_int(Ln) = E_complex(Ln) - E_ion(Ln3+) - E_cage

evaluated at a **fixed** geometry, so E_cage is bit-identical across the series
and contributes only a constant offset.  The *shape* of E_int against Z is then
a pure bonding observable.  Alongside it we take the metal Mulliken charge and
the HOMO-LUMO gap, which need no reference at all.

Open-shell states
-----------------
GFN2 puts f in the core, so ``--uhf 0`` is correct there and is what every
published re-optimisation used.  Under g-xTB the f electrons are explicit, so
Gd(III) is f7 with **seven** unpaired electrons and running it closed-shell
forces an unphysical state.  Since the entire question is whether f-shell
physics shows up, getting this wrong would manufacture a null.  ``high_spin_uhf``
below is the Hund's-rule count for Ln(III).

Solvation
---------
g-xTB has **no ALPB/GBSA parameters** -- the flag the production pipeline uses
is a hard error.  It does support ddCOSMO via ``--cosmo``.  Note ``--cpcmx`` is
accepted and then *silently ignored* (energy bit-identical to gas phase), which
is exactly the sort of thing that produces gas-phase numbers labelled solvated.
This module therefore states the solvation model in every record.

    python3 -m automl.qc.gxtb_probe --smoke
    python3 -m automl.qc.gxtb_probe --series --anchor <xyz> --method both
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.qc.xtb_backend import (  # noqa: E402
    HARTREE_EV, _read_mulliken, read_plain_xyz, write_plain_xyz, xtb_env,
)

GXTB_BIN = Path.home() / "opt/xtb-6.7.1/bin/xtb"
OUT = _REPO / "automl/artifacts/gxtb_probe"

# Ln(III): f-count is Z-57, unpaired electrons by Hund's first rule.
LANTHANIDES = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
               "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Z_OF = {s: 57 + i for i, s in enumerate(LANTHANIDES)}


def f_count(sym: str) -> int:
    """Number of 4f electrons in the +3 ion."""
    return Z_OF[sym] - 57


def high_spin_uhf(sym: str) -> int:
    """Unpaired electrons in high-spin Ln(III), i.e. N_alpha - N_beta."""
    n = f_count(sym)
    return n if n <= 7 else 14 - n


def _run(args: list[str], wd: Path, threads: int, timeout: int
         ) -> tuple[int, str]:
    proc = subprocess.run([str(GXTB_BIN)] + args, cwd=wd,
                          env=xtb_env(GXTB_BIN, threads),
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def _method_args(method: str) -> list[str]:
    if method == "gxtb":
        return ["--gxtb"]
    if method == "gfn2":
        return ["--gfn", "2"]
    raise ValueError(f"unknown method {method!r}")


def _solvent_args(method: str, solvent: str | None) -> list[str]:
    """ALPB for GFN2, ddCOSMO for g-xTB.  Never --cpcmx: it is a silent no-op."""
    if not solvent:
        return []
    return ["--cosmo", solvent] if method == "gxtb" else ["--alpb", solvent]


def _grep_gap(text: str) -> float | None:
    for line in text.splitlines():
        if "HOMO-LUMO GAP" in line.upper():
            for tok in line.split():
                try:
                    return float(tok)
                except ValueError:
                    continue
    return None


# Energy parsing is the proven regex from ``xtb_backend`` rather than a fresh
# token scan: the g-xTB log prints "total energy" in the SCF trace as well as
# "TOTAL ENERGY" in the summary block, and a looser parser picks up the wrong
# one on a run that fails late.
from automl.qc.xtb_backend import _grep_energy  # noqa: E402,F401


def single_point(symbols, coords, *, charge: int, uhf: int, method: str,
                 solvent: str | None = None, threads: int = 1,
                 timeout: int = 3600) -> dict[str, Any]:
    """One SCF.  Returns energy, gap, and per-atom Mulliken charges."""
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        write_plain_xyz(symbols, coords, wd / "in.xyz")
        args = (["in.xyz"] + _method_args(method)
                + ["--chrg", str(charge), "--uhf", str(uhf), "--sp",
                   "--norestart"]
                + _solvent_args(method, solvent))
        try:
            rc, out = _run(args, wd, threads, timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout", "method": method,
                    "seconds": time.time() - t0}
        e = _grep_energy(out)
        scf_bad = "did not converge" in out or "SCF not converged" in out
        rec: dict[str, Any] = {
            "ok": rc == 0 and e is not None,
            "reason": None if (rc == 0 and e is not None)
                      else ("scf_not_converged" if scf_bad else f"rc_{rc}"),
            "method": method, "solvent": solvent, "charge": charge, "uhf": uhf,
            "energy_eh": e,
            "energy_ev": None if e is None else e * HARTREE_EV,
            "homo_lumo_gap_ev": _grep_gap(out),
            "seconds": time.time() - t0,
        }
        if not rec["ok"]:
            rec["log_tail"] = out[-3000:]
        q = _read_mulliken(wd, len(symbols))
        if q is not None:
            rec["partial_charges"] = [float(x) for x in q]
        return rec


def optimize(symbols, coords, *, charge: int, uhf: int, method: str,
             solvent: str | None = None, opt_level: str = "tight",
             maxcycle: int = 750, threads: int = 1, timeout: int = 7200,
             etemp: float | None = None) -> dict[str, Any]:
    """Relax a substituted structure and report the M-donor shell.

    The electronic probe above is at *fixed* geometry, which isolates the
    electronic structure but says nothing about the coordinates a model would
    actually be given.  This is the arm that matters for the project: if the
    optimised M-donor response carries f-shell structure beyond linear-in-Z,
    then re-optimising the set with g-xTB puts information into the geometry
    that GFN2 cannot represent at all.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        write_plain_xyz(symbols, coords, wd / "in.xyz")
        (wd / "xtb.inp").write_text(f"$opt\n   maxcycle={maxcycle}\n$end\n")
        args = (["in.xyz"] + _method_args(method)
                + ["--chrg", str(charge), "--uhf", str(uhf), "--opt", opt_level,
                   "--input", "xtb.inp", "--norestart"]
                + _solvent_args(method, solvent))
        if etemp is not None:
            args += ["--etemp", str(etemp)]
        try:
            rc, out = _run(args, wd, threads, timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout", "method": method,
                    "seconds": time.time() - t0}
        f = wd / "xtbopt.xyz"
        if rc != 0 or not f.exists():
            scf_bad = "did not converge" in out or "SCF not converged" in out
            return {"ok": False,
                    "reason": "scf_not_converged" if scf_bad else f"rc_{rc}",
                    "method": method, "etemp": etemp,
                    "seconds": time.time() - t0, "log_tail": out[-3000:]}
        so, xo = read_plain_xyz(f)
        rmsd = float(np.sqrt(((np.asarray(xo) - np.asarray(coords)) ** 2)
                             .sum(axis=1).mean()))
        return {"ok": True, "method": method, "solvent": solvent, "uhf": uhf,
                "charge": charge, "etemp": etemp,
                "symbols": so, "coords": [[float(v) for v in r] for r in xo],
                "energy_eh": _grep_energy(out),
                "homo_lumo_gap_ev": _grep_gap(out),
                "converged": "GEOMETRY OPTIMIZATION CONVERGED" in out.upper(),
                "rmsd_from_input_ang": rmsd,
                "seconds": time.time() - t0}


def optimize_with_retry(symbols, coords, **kw) -> dict[str, Any]:
    """Retry a marginal SCF with electronic smearing, and say so in the record.

    One structure in fifteen (Nd) oscillated at the 1e-5 Eh level and hit the
    250-cycle wall, then converged on an identical rerun -- it sits exactly on
    the tolerance.  Smearing is the standard xtb remedy, but it perturbs the
    electronic structure, so any structure that needed it is flagged rather
    than silently pooled with the rest.
    """
    r = optimize(symbols, coords, **kw)
    if r.get("ok") or r.get("reason") != "scf_not_converged":
        return r
    for et in (300.0, 1000.0):
        kw2 = dict(kw, etemp=et)
        r2 = optimize(symbols, coords, **kw2)
        if r2.get("ok"):
            r2["needed_etemp"] = et
            return r2
    return r


def free_ion(sym: str, *, method: str, solvent: str | None = None,
             threads: int = 1) -> dict[str, Any]:
    """E(Ln3+) as an isolated ion -- the atomic reference E_int subtracts off."""
    return single_point([sym], np.zeros((1, 3)), charge=3,
                        uhf=high_spin_uhf(sym), method=method,
                        solvent=solvent, threads=threads)


def substitute_metal(symbols: list[str], new_metal: str) -> list[str]:
    """Swap the lanthanide token.  Coordinates are untouched by construction."""
    out, seen = [], 0
    for s in symbols:
        if s in Z_OF:
            out.append(new_metal)
            seen += 1
        else:
            out.append(s)
    if seen != 1:
        raise ValueError(f"expected exactly 1 lanthanide, found {seen}")
    return out


def metal_index(symbols: list[str]) -> int:
    for i, s in enumerate(symbols):
        if s in Z_OF:
            return i
    raise ValueError("no lanthanide in structure")


def donor_distances(symbols, coords, cn: int = 9) -> np.ndarray:
    """Distances to the ``cn`` nearest non-hydrogen atoms -- the M-donor shell."""
    mi = metal_index(list(symbols))
    xyz = np.asarray(coords, dtype=float)
    d = np.linalg.norm(xyz - xyz[mi], axis=1)
    mask = np.array([s != "H" for s in symbols])
    mask[mi] = False
    cand = np.where(mask)[0]
    return np.sort(d[cand])[:cn]


def series(anchor: Path, *, method: str, charge: int = 3,
           solvent: str | None = None, metals: list[str] | None = None,
           threads: int = 1, do_opt: bool = False) -> list[dict[str, Any]]:
    """Substitute every lanthanide into one fixed anchor geometry and probe it.

    Fixed geometry is the point: it holds the ligand cage bit-identical across
    the series, so every difference between records is the metal's electronic
    structure and nothing else.
    """
    sym0, xyz0 = read_plain_xyz(anchor)
    metals = metals or LANTHANIDES
    recs = []
    for m in metals:
        sym = substitute_metal(list(sym0), m)
        u = high_spin_uhf(m) if method == "gxtb" else 0
        r = single_point(sym, xyz0, charge=charge, uhf=u, method=method,
                         solvent=solvent, threads=threads)
        r.update({"metal": m, "z": Z_OF[m], "f_count": f_count(m),
                  "unpaired": high_spin_uhf(m), "anchor": str(anchor),
                  "n_atoms": len(sym), "mode": "single_point"})
        if r.get("ok") and r.get("partial_charges"):
            r["metal_mulliken"] = r["partial_charges"][metal_index(sym)]
        recs.append(r)
        print(f"  {m} Z={Z_OF[m]} f{f_count(m)} uhf={u}: "
              f"{'ok' if r['ok'] else r['reason']} "
              f"E={r.get('energy_eh')} gap={r.get('homo_lumo_gap_ev')} "
              f"{r['seconds']:.1f}s", flush=True)
    return recs


def interaction_energies(recs: list[dict], ion: dict[str, float]
                         ) -> list[dict[str, Any]]:
    """E_int = E_complex - E_ion, up to the constant cage term.

    The cage energy is identical for every metal at fixed geometry, so it is an
    additive constant and cannot affect any statement about *shape*.  It is not
    subtracted here precisely so that no fragment calculation can go wrong
    silently; every downstream test is invariant to a constant.
    """
    out = []
    for r in recs:
        if not r.get("ok") or r["metal"] not in ion:
            continue
        out.append({"metal": r["metal"], "z": r["z"], "f_count": r["f_count"],
                    "unpaired": r["unpaired"],
                    "e_complex_eh": r["energy_eh"],
                    "e_ion_eh": ion[r["metal"]],
                    "e_int_eh": r["energy_eh"] - ion[r["metal"]],
                    "metal_mulliken": r.get("metal_mulliken"),
                    "homo_lumo_gap_ev": r.get("homo_lumo_gap_ev")})
    return out


def linearity(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """How much of y is NOT a straight line in x.

    This is the C-I statistic.  Under GFN2 the answer must be ~0 by
    construction; the residual is the machine precision of the parameter file.
    A large structured residual under g-xTB is f-shell physics.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return {"n": len(x)}
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    span = float(np.ptp(y))
    return {"n": int(len(x)), "slope": float(coef[0]),
            "max_abs_resid": float(np.max(np.abs(resid))),
            "rms_resid": float(np.sqrt(np.mean(resid ** 2))),
            "span": span,
            "resid_frac_of_span": float(np.max(np.abs(resid)) / span)
                                  if span > 0 else float("nan")}


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", type=Path)
    ap.add_argument("--method", default="both",
                    choices=("gxtb", "gfn2", "both"))
    ap.add_argument("--charge", type=int, default=3)
    ap.add_argument("--solvent", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--smoke", action="store_true",
                    help="3 metals only, to check convergence and timing")
    args = ap.parse_args()

    if not GXTB_BIN.is_file():
        raise SystemExit(f"g-xTB binary not found at {GXTB_BIN}")
    if args.anchor is None:
        raise SystemExit("give --anchor <xyz>")

    metals = ["Nd", "Gd", "Er"] if args.smoke else None
    methods = ["gfn2", "gxtb"] if args.method == "both" else [args.method]
    result: dict[str, Any] = {"anchor": str(args.anchor), "tag": args.tag,
                              "solvent": args.solvent, "charge": args.charge,
                              "smoke": args.smoke, "arms": {}}
    for meth in methods:
        print(f"[{meth}] series on {args.anchor.name}", flush=True)
        recs = series(args.anchor, method=meth, charge=args.charge,
                      solvent=args.solvent, metals=metals,
                      threads=args.threads)
        print(f"[{meth}] free ions", flush=True)
        ion = {}
        for m in (metals or LANTHANIDES):
            r = free_ion(m, method=meth, solvent=args.solvent,
                         threads=args.threads)
            if r.get("ok"):
                ion[m] = r["energy_eh"]
            print(f"  ion {m}: {'ok' if r['ok'] else r['reason']} "
                  f"{r.get('energy_eh')}", flush=True)
        ei = interaction_energies(recs, ion)
        arm: dict[str, Any] = {"records": recs, "ion_energy_eh": ion,
                               "interaction": ei,
                               "n_ok": sum(1 for r in recs if r.get("ok")),
                               "n_total": len(recs)}
        if len(ei) >= 3:
            z = np.array([e["z"] for e in ei], float)
            arm["linearity_e_int"] = linearity(z, [e["e_int_eh"] for e in ei])
            qs = [e["metal_mulliken"] for e in ei]
            if all(q is not None for q in qs):
                arm["linearity_metal_charge"] = linearity(z, qs)
        result["arms"][meth] = arm

    _dump(result, OUT / f"{args.tag}.json")
    print(f"\n[gxtb_probe] wrote {OUT / (args.tag + '.json')}")
    for meth, arm in result["arms"].items():
        print(f"  {meth}: {arm['n_ok']}/{arm['n_total']} converged")
        for key in ("linearity_e_int", "linearity_metal_charge"):
            if key in arm:
                s = arm[key]
                print(f"    {key}: span={s['span']:.6f} "
                      f"max|resid|={s['max_abs_resid']:.3e} "
                      f"({s['resid_frac_of_span']:.4%} of span)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
