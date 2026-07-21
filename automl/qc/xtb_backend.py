#!/usr/bin/env python3
"""GFN2-xTB backend: single points and geometry optimisation.

Why this exists
---------------
Every geometry shipped in ``data/geometries`` stopped on a loose ``fmax = 0.2
eV/A`` criterion -- the residual-force distribution is hard-capped at 0.19999
with 94 % of structures between 0.15 and 0.20, i.e. the optimiser terminated
*on the threshold* rather than because the structure relaxed.  That is 4x
looser than ASE's default and ~10x looser than tight practice, and it means
some of the descriptor scatter previously attributed to conformational
diversity is really optimisation noise.  This module re-optimises properly.

Chemistry is preserved, not re-decided
--------------------------------------
The molecular charge is **read per structure**, never assumed.  xTB Mulliken
populations sum to the molecular charge, so ``round(nansum(partial_charges))``
recovers exactly the charge the original calculation used.  This matters: the
charge is *not* uniform across the set --

    +3  1214 files      [Ln(L)n]3+ with neutral ligands
    +2   241 files      one deprotonated / anionic ligand bound
    +1    68 files      two anionic ligands bound

so a blanket ``--chrg 3`` would silently mis-specify 309 of 1523 structures
(20 %).  Three files carry no xTB properties at all and cannot have their
charge inferred; they are reported and skipped rather than guessed.

Spin is closed-shell (``uhf = 0``): GFN2-xTB carries lanthanide f-electrons in
the core, and every stored geometry has ``initial_magmoms`` summing to zero.

Element composition, atom count and connectivity come from the input file
unchanged -- ``plan_complex()`` decisions are never revisited here.

Backend
-------
Prefers the static ``xtb`` binary (ANCopt optimiser, mature ALPB solvation).
Set ``XTB_BIN`` to override discovery.  ``tblite`` is installed as a fallback
for single points but is not used for optimisation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.geometry_features import read_extxyz  # noqa: E402

HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903
# 1 Eh/bohr in eV/A -- used to translate xtb's gradient thresholds into the
# eV/A units the existing descriptors and diagnostics are expressed in.
EH_BOHR_TO_EV_ANG = HARTREE_EV / BOHR_ANG  # 51.42208...

# xtb ANCopt named levels -> gradient convergence in Eh/bohr, and the eV/A
# equivalent.  The shipped geometries sit at 0.2 eV/A, i.e. looser than every
# level here.
OPT_LEVELS = {
    "loose":  (2.0e-3, 2.0e-3 * EH_BOHR_TO_EV_ANG),   # 0.103 eV/A
    "normal": (1.0e-3, 1.0e-3 * EH_BOHR_TO_EV_ANG),   # 0.051 eV/A
    "tight":  (8.0e-4, 8.0e-4 * EH_BOHR_TO_EV_ANG),   # 0.041 eV/A
    "vtight": (2.0e-4, 2.0e-4 * EH_BOHR_TO_EV_ANG),   # 0.010 eV/A
}

DEFAULT_UHF = 0
FALLBACK_CHARGE = 3   # only used when charges are missing, and always flagged


def infer_charge(geometry) -> tuple[int | None, str]:
    """Recover the molecular charge from stored xTB Mulliken populations.

    Mulliken populations sum to the molecular charge, so the rounded sum is the
    charge the original calculation ran at.  Returns ``(charge, provenance)``;
    ``charge`` is None when the structure carries no xTB properties, so the
    caller can skip it rather than guess.
    """
    q = np.asarray(geometry.partial_charges, dtype=float)
    if not np.isfinite(q).any():
        return None, "missing_xtb_properties"
    total = float(np.nansum(q))
    charge = int(round(total))
    # A genuine Mulliken sum lands within numerical noise of an integer; a large
    # residual means the file is not what we think it is.
    if abs(total - charge) > 0.05:
        return None, f"non_integer_charge_sum_{total:.3f}"
    return charge, "mulliken_sum"


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------
def find_xtb() -> Path | None:
    """Locate the xtb binary: $XTB_BIN, then ~/opt, then $PATH."""
    env = os.environ.get("XTB_BIN")
    if env and Path(env).is_file():
        return Path(env)
    for cand in (Path.home() / "opt/xtb-dist/bin/xtb",
                 Path.home() / "opt/xtb/bin/xtb"):
        if cand.is_file():
            return cand
    which = shutil.which("xtb")
    return Path(which) if which else None


def xtb_env(binary: Path, threads: int = 1) -> dict[str, str]:
    """Environment for a standalone xtb tree (needs XTBPATH and lib on the path)."""
    home = binary.parent.parent
    env = dict(os.environ)
    env["XTBHOME"] = str(home)
    env["XTBPATH"] = str(home / "share/xtb")
    env["LD_LIBRARY_PATH"] = f"{home / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    # xtb parallelises poorly beyond a few threads and we run many structures
    # concurrently, so one thread per structure is the efficient choice.
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OMP_STACKSIZE"] = "4G"
    return env


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def write_plain_xyz(symbols, coords, path: Path, comment: str = "") -> None:
    lines = [str(len(symbols)), comment]
    for s, (x, y, z) in zip(symbols, coords):
        lines.append(f"{s:2s} {x:20.12f} {y:20.12f} {z:20.12f}")
    path.write_text("\n".join(lines) + "\n")


def read_plain_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    sym, xyz = [], []
    for ln in lines[2:2 + n]:
        v = ln.split()
        sym.append(v[0])
        xyz.append([float(v[1]), float(v[2]), float(v[3])])
    return sym, np.asarray(xyz, dtype=float)


def _grep_energy(text: str) -> float | None:
    m = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", text)
    if m:
        return float(m[-1])
    m = re.findall(r"\|\s*TOTAL ENERGY\s+(-?\d+\.\d+)", text)
    return float(m[-1]) if m else None


def _grep_gradient_norm(text: str) -> float | None:
    m = re.findall(r"GRADIENT NORM\s+(\d+\.\d+)\s+Eh", text)
    return float(m[-1]) if m else None


def _grep_converged(text: str) -> bool:
    return ("GEOMETRY OPTIMIZATION CONVERGED" in text
            or "*** GEOMETRY OPTIMIZATION CONVERGED" in text)


def _grep_cycles(text: str) -> int | None:
    m = re.findall(r"optimization cycle\s+(\d+)", text, flags=re.I)
    return int(m[-1]) if m else None


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _run(binary: Path, args: list[str], workdir: Path, threads: int,
         timeout: int) -> tuple[int, str]:
    proc = subprocess.run([str(binary)] + args, cwd=workdir,
                          env=xtb_env(binary, threads), capture_output=True,
                          text=True, timeout=timeout)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def single_point(symbols, coords, *, charge: int,
                 uhf: int = DEFAULT_UHF, solvent: str | None = None,
                 binary: Path | None = None, threads: int = 1,
                 timeout: int = 3600) -> dict[str, Any]:
    """GFN2-xTB single point.  Returns energy in Eh and eV."""
    binary = binary or find_xtb()
    if binary is None:
        raise RuntimeError("no xtb binary found; set XTB_BIN")
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        write_plain_xyz(symbols, coords, wd / "in.xyz")
        args = ["in.xyz", "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf),
                "--sp", "--norestart"]
        if solvent:
            args += ["--alpb", solvent]
        rc, out = _run(binary, args, wd, threads, timeout)
        e = _grep_energy(out)
        return {"ok": rc == 0 and e is not None, "returncode": rc,
                "energy_eh": e, "energy_ev": None if e is None else e * HARTREE_EV,
                "gradient_norm_eh_bohr": _grep_gradient_norm(out),
                "log_tail": out[-3000:] if rc != 0 else ""}


def optimize(symbols, coords, *, charge: int,
             uhf: int = DEFAULT_UHF, solvent: str | None = None,
             opt_level: str = "tight", maxcycle: int = 750,
             binary: Path | None = None, threads: int = 1,
             timeout: int = 21600, etemp: float | None = None) -> dict[str, Any]:
    """GFN2-xTB geometry optimisation (ANCopt).

    Returns the relaxed coordinates plus an explicit convergence record: the
    achieved max/RMS force in eV/A, whether xtb declared convergence, cycles
    used, and the RMSD from the input structure.  Nothing is reported as
    converged unless xtb said so *and* the force check agrees.
    """
    binary = binary or find_xtb()
    if binary is None:
        raise RuntimeError("no xtb binary found; set XTB_BIN")
    if opt_level not in OPT_LEVELS:
        raise ValueError(f"unknown opt_level {opt_level!r}")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        write_plain_xyz(symbols, coords, wd / "in.xyz")
        (wd / "xtb.inp").write_text(f"$opt\n   maxcycle={maxcycle}\n$end\n")
        args = ["in.xyz", "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf),
                "--opt", opt_level, "--input", "xtb.inp", "--norestart"]
        if solvent:
            args += ["--alpb", solvent]
        if etemp is not None:
            # Fractional-occupation smearing.  86 octanol optimisations aborted
            # with "scf: Self consistent charge iterator did not converge" --
            # not a timeout (a 3x longer limit changed nothing) and not memory.
            # The failures concentrate on Eu (26 of 86), whose near-degenerate
            # frontier orbitals are the classic hard-SCF case.  Raising the
            # electronic temperature smears the occupations and is the standard
            # xtb remedy.  It perturbs the electronic structure slightly, so
            # every structure that needs it is recorded rather than silently
            # mixed in with the rest.
            args += ["--etemp", str(etemp)]
        try:
            rc, out = _run(binary, args, wd, threads, timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout", "seconds": time.time() - t0,
                    "opt_level": opt_level, "solvent": solvent}
        opt_file = wd / "xtbopt.xyz"
        if rc != 0 or not opt_file.exists():
            scf_failed = "did not converge" in out or "SCF not converged" in out
            return {"ok": False,
                    "reason": ("scf_not_converged" if scf_failed
                               else f"xtb_rc_{rc}"),
                    "seconds": time.time() - t0,
                    "opt_level": opt_level, "solvent": solvent,
                    "etemp": etemp, "log_tail": out[-3000:]}
        sym_out, xyz_out = read_plain_xyz(opt_file)

        # Independent force check: a single point on the relaxed structure with
        # --grad gives the true residual gradient, which is what the descriptor
        # pipeline reports in eV/A.  Trusting xtb's own "converged" banner alone
        # is how the current dataset ended up capped at 0.2 eV/A.
        grad = _residual_forces(binary, sym_out, xyz_out, charge, uhf, solvent,
                                threads, timeout=min(timeout, 3600))
        rmsd = float(np.sqrt(((np.asarray(xyz_out) - np.asarray(coords)) ** 2)
                             .sum(axis=1).mean()))
        target_ev_ang = OPT_LEVELS[opt_level][1]
        return {
            "ok": True,
            "symbols": sym_out,
            "coords": xyz_out,
            "energy_eh": _grep_energy(out),
            "energy_ev": (lambda e: None if e is None else e * HARTREE_EV)(_grep_energy(out)),
            "xtb_converged": _grep_converged(out),
            "cycles": _grep_cycles(out),
            "force_max_ev_ang": grad.get("force_max_ev_ang"),
            "force_rms_ev_ang": grad.get("force_rms_ev_ang"),
            "target_force_ev_ang": target_ev_ang,
            "meets_target": (grad.get("force_max_ev_ang") is not None
                             and grad["force_max_ev_ang"] <= target_ev_ang * 1.5),
            "rmsd_from_input_ang": rmsd,
            "opt_level": opt_level,
            "solvent": solvent,
            "charge": charge,
            "uhf": uhf,
            "seconds": time.time() - t0,
        }


def _residual_forces(binary, symbols, coords, charge, uhf, solvent, threads,
                     timeout) -> dict[str, Any]:
    """Max/RMS residual force in eV/A from a --grad single point."""
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        write_plain_xyz(symbols, coords, wd / "g.xyz")
        args = ["g.xyz", "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf),
                "--grad", "--norestart"]
        if solvent:
            args += ["--alpb", solvent]
        try:
            rc, out = _run(binary, args, wd, threads, timeout)
        except subprocess.TimeoutExpired:
            return {}
        gfile = wd / "gradient"
        if not gfile.exists():
            return {}
        # Turbomole 'gradient' format: coordinates block then gradient block,
        # both natoms lines; gradients are Eh/bohr, printed in Fortran D notation.
        rows = [ln for ln in gfile.read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("$")]
        n = len(symbols)
        gtxt = rows[-n:] if len(rows) >= n else []
        try:
            g = np.array([[float(x.replace("D", "E").replace("d", "E"))
                           for x in ln.split()[:3]] for ln in gtxt])
        except ValueError:
            return {}
        if g.shape != (n, 3):
            return {}
        fn = np.linalg.norm(g, axis=1) * EH_BOHR_TO_EV_ANG
        return {"force_max_ev_ang": float(fn.max()),
                "force_rms_ev_ang": float(np.sqrt((fn ** 2).mean()))}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def selftest(limit: int = 3) -> int:
    """Reproduce stored GFN2-xTB energies using per-structure inferred charges.

    Deliberately samples across the charge states present in the set (+3, +2,
    +1) rather than only the common one, because a blanket charge would still
    look correct on a +3-only sample.  Structures with no stored energy are
    reported separately -- they are not failures, they simply cannot be
    validated this way.
    """
    binary = find_xtb()
    print(f"xtb binary : {binary}")
    if binary is None:
        print("FAIL: no xtb binary"); return 1
    _, ver = _run(binary, ["--version"], Path.cwd(), 1, 120)
    print("xtb version:", next((l.strip() for l in ver.splitlines()
                                if "version" in l.lower()), "?"))
    print("convergence levels (Eh/bohr -> eV/A): "
          + ", ".join(f"{k}={v[1]:.3f}" for k, v in OPT_LEVELS.items()))
    print("shipped geometries all sit at ~0.20 eV/A\n")

    xyzs = sorted(_REPO_ROOT.glob("data/geometries/*/*.xyz"),
                  key=lambda p: p.stat().st_size)
    # bucket by inferred charge so the test spans +3/+2/+1
    buckets: dict[int, list] = {}
    skipped = []
    for p in xyzs:
        try:
            g = read_extxyz(p)
        except Exception:
            continue
        c, why = infer_charge(g)
        if c is None:
            skipped.append((p.name, why)); continue
        if not np.isfinite(g.energy_eV):
            skipped.append((p.name, "no_stored_energy")); continue
        buckets.setdefault(c, []).append((p, g, c))

    print(f"charge states found: {sorted(buckets)}  "
          + "  ".join(f"q={k}:{len(v)}" for k, v in sorted(buckets.items())))
    print(f"structures not validatable: {len(skipped)}\n")

    failures = tested = 0
    for c in sorted(buckets, reverse=True):
        for p, g, charge in buckets[c][:limit]:
            res = single_point(g.symbols, g.coordinates, charge=charge,
                               uhf=DEFAULT_UHF, binary=binary, threads=1)
            tested += 1
            if not res["ok"]:
                print(f"  q={charge:+d} {p.name[:44]:46s} FAIL rc={res['returncode']}")
                failures += 1
                continue
            d_eh = (res["energy_ev"] - g.energy_eV) / HARTREE_EV
            ok = abs(d_eh) < 1e-4
            failures += (not ok)
            print(f"  q={charge:+d} {p.name[:44]:46s} n={len(g.symbols):3d} "
                  f"d={d_eh:+.2e} Eh  {'OK' if ok else 'MISMATCH'}")

    # A blanket charge=+3 must visibly fail on the non-+3 structures, otherwise
    # this test proves nothing about the inference.
    print("\n  control: forcing charge=+3 on a non-+3 structure should MISMATCH")
    ctrl = next((v[0] for k, v in sorted(buckets.items()) if k != 3), None)
    if ctrl is not None:
        p, g, true_c = ctrl
        r = single_point(g.symbols, g.coordinates, charge=3, uhf=DEFAULT_UHF,
                         binary=binary, threads=1)
        if r["ok"]:
            d = (r["energy_ev"] - g.energy_eV) / HARTREE_EV
            good = abs(d) > 1e-3
            print(f"    true q={true_c:+d}, forced q=+3 -> d={d:+.3e} Eh "
                  f"{'(control OK: inference matters)' if good else '(CONTROL FAILED)'}")
            failures += (not good)
    print()
    if skipped:
        print(f"not validatable ({len(skipped)}): "
              + ", ".join(f"{n[:34]}[{w}]" for n, w in skipped[:4]))
    if failures:
        print(f"\nSELFTEST FAILED ({failures}/{tested + 1}) -- do NOT run Stage 1.")
    else:
        print(f"\nSELFTEST PASSED ({tested} structures across "
              f"{len(buckets)} charge states) -- per-structure charge inference "
              f"reproduces the stored energies.")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=2)
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.limit)
    b = find_xtb()
    print(json.dumps({"xtb_binary": str(b) if b else None,
                      "opt_levels_ev_ang": {k: v[1] for k, v in OPT_LEVELS.items()},
                      "charge": DEFAULT_CHARGE, "uhf": DEFAULT_UHF}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
