#!/usr/bin/env python3
"""Does the *optimised geometry* carry f-shell structure, or only the wavefunction?

The fixed-geometry probe (``gxtb_probe``) answered the electronic question and
answered it loudly: after removing the linear-in-Z trend, GFN2's HOMO-LUMO gap
has **0.00075 eV** of residual structure across Ce->Lu, while g-xTB has
**0.28 eV** -- 370x more -- and that residual reproduces at r = +0.97 between
two independent runs, so it is physics and not SCF noise.  g-xTB shows a
**+1.15 eV** discontinuity at the half-filled f7 shell (the gadolinium break);
GFN2 shows +0.012 eV, which is not a break at all but the trivial consequence
of splitting a straight monotone ramp in the middle.

None of that is yet worth anything to this project, because our models are not
given wavefunctions.  They are given **coordinates**.  The question this module
settles is whether the structure survives into the relaxed geometry:

    is the optimised M-donor response to lanthanide identity
    more than one linear-in-Z scalar under g-xTB?

Under GFN2 the answer is provably no -- every lanthanide parameter is linear
interpolation between the Ce and Lu anchors, so the relaxed geometry can carry
exactly one scalar of metal identity.  That is the rank-1 ceiling, and it is why
eight different 3D encoders were mutually interchangeable at effective rank 1.05.

Design
------
Both arms are run under **one binary, one protocol, one set of anchors**; the
only difference is the Hamiltonian.  Several ligand families are used because a
break seen in one complex could be a property of that complex.

``--uhf 0`` is correct for GFN2 (f is in the core; every published
re-optimisation used it) and *wrong* for g-xTB, where f is in the valence and
Gd(III) is f7 with seven unpaired electrons.  The closed-shell g-xTB arm is
therefore run as a deliberate control: if the geometric break is f-shell
physics, forcing the closed-shell state should damage it.

    python3 -m automl.qc.gxtb_series --anchors 6 --workers 48
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.qc.gxtb_probe import (  # noqa: E402
    LANTHANIDES, Z_OF, donor_distances, f_count, high_spin_uhf, metal_index,
    optimize_with_retry, substitute_metal,
)
from automl.qc.xtb_backend import read_plain_xyz  # noqa: E402

SERIAL = _REPO / "automl/artifacts/serial_metals"
OUT = _REPO / "automl/artifacts/gxtb_series"


def pick_anchors(n: int, max_atoms: int | None = None) -> list[dict[str, Any]]:
    """One anchor per ligand family, chosen for diversity of ligand and CN.

    Anchors are *already-relaxed* structures.  Starting from a converged
    minimum rather than a shipped start is the same design point that made the
    serial construction work: re-optimising from a shipped start travels a
    median 1.87 A and can bifurcate into another basin, which is precisely the
    conformer noise this whole line of work exists to remove.
    """
    recs = []
    for p in (SERIAL / "records").glob("serial__*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:                                          # noqa: BLE001
            continue
        if r.get("ok") and r.get("path") and r.get("mode") == "serial":
            recs.append(r)
    fam: dict[str, list[dict]] = {}
    for r in recs:
        fam.setdefault(r.get("family", ""), []).append(r)
    # Prefer families that are large (well-sampled) and small in atom count
    # (cheap), and take at most one anchor per distinct ligand so the arms are
    # not six near-copies of one molecule.
    seen_lig: set[str] = set()
    out = []
    for key in sorted(fam, key=lambda k: (-len(fam[k]),
                                          min(x.get("n_atoms", 1e9)
                                              for x in fam[k]))):
        lig = key.split("||")[0]
        if lig in seen_lig:
            continue
        member = min(fam[key], key=lambda x: x.get("n_atoms", 1e9))
        if max_atoms is not None and member.get("n_atoms", 0) > max_atoms:
            continue
        seen_lig.add(lig)
        out.append({"family": key, "path": member["path"],
                    "charge": int(member.get("charge", 3)),
                    "n_atoms": int(member.get("n_atoms", 0)),
                    "cn": member.get("cn"), "anchor_metal": member.get("metal")})
        if len(out) >= n:
            break
    return out


def _one(task: dict[str, Any]) -> dict[str, Any]:
    sym0, xyz0 = read_plain_xyz(_REPO / task["path"])
    sym = substitute_metal(list(sym0), task["metal"])
    uhf = (high_spin_uhf(task["metal"]) if task["arm"] == "gxtb_hs" else 0)
    method = "gfn2" if task["arm"] == "gfn2" else "gxtb"
    r = optimize_with_retry(sym, xyz0, charge=task["charge"], uhf=uhf,
                            method=method, solvent=task.get("solvent"),
                            threads=1, timeout=task.get("timeout", 7200))
    rec = {k: task[k] for k in ("family", "path", "metal", "arm", "charge")}
    rec.update({"z": Z_OF[task["metal"]], "f_count": f_count(task["metal"]),
                "uhf": uhf, "ok": bool(r.get("ok")),
                "reason": r.get("reason"), "seconds": r.get("seconds"),
                "needed_etemp": r.get("needed_etemp"),
                "energy_eh": r.get("energy_eh"),
                "homo_lumo_gap_ev": r.get("homo_lumo_gap_ev"),
                "converged": r.get("converged"),
                "rmsd_from_input_ang": r.get("rmsd_from_input_ang")})
    if r.get("ok"):
        cn = int(task.get("cn") or 9)
        d = donor_distances(r["symbols"], np.asarray(r["coords"]), cn=cn)
        rec["donor_distances"] = [float(x) for x in d]
        rec["mean_m_donor"] = float(np.mean(d))
        rec["coords"] = r["coords"]
        rec["symbols"] = r["symbols"]
        rec["metal_index"] = metal_index(r["symbols"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--solvent", default=None)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--arms", default="gfn2,gxtb_hs,gxtb_cs")
    ap.add_argument("--tag", default="opt")
    ap.add_argument("--max-atoms", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    anchors = pick_anchors(args.anchors, max_atoms=args.max_atoms)
    arms = [a for a in args.arms.split(",") if a]
    # Shard BY ANCHOR, never by task: a compliance coefficient is a fit across
    # the whole 15-metal series, so a family split across jobs could come back
    # half-finished and be silently fitted on a partial series.
    if args.num_shards > 1:
        anchors = [a for i, a in enumerate(anchors)
                   if i % args.num_shards == args.shard]
    tasks = [dict(a, metal=m, arm=arm, solvent=args.solvent,
                  timeout=args.timeout)
             for a in anchors for m in LANTHANIDES for arm in arms]
    # Cheapest first, so a job that runs out of wall clock loses the fewest
    # complete series rather than a random subset.
    tasks.sort(key=lambda t: t.get("n_atoms", 0))
    print(f"[gxtb_series] {len(anchors)} anchors x {len(LANTHANIDES)} metals "
          f"x {len(arms)} arms = {len(tasks)} optimisations", flush=True)
    for a in anchors:
        print(f"   {a['n_atoms']:4d} atoms CN={a['cn']} {a['family'][:70]}",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    done, n_ok = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:                                 # noqa: BLE001
                t = futs[fu]
                r = {"ok": False, "reason": f"exception:{e}",
                     "metal": t["metal"], "arm": t["arm"],
                     "family": t["family"]}
            done.append(r)
            n_ok += bool(r.get("ok"))
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] ok={n_ok}", flush=True)

    # Coordinates are bulky and are what makes this resumable; keep them in a
    # sidecar so the summary stays readable.
    slim = [{k: v for k, v in r.items() if k not in ("coords", "symbols")}
            for r in done]
    tmp = OUT / f"{args.tag}.json.tmp"
    tmp.write_text(json.dumps({"anchors": anchors, "arms": arms,
                               "solvent": args.solvent, "records": slim},
                              indent=2, default=float) + "\n")
    tmp.replace(OUT / f"{args.tag}.json")
    tmp2 = OUT / f"{args.tag}_coords.json.tmp"
    tmp2.write_text(json.dumps(
        [{k: r[k] for k in ("family", "metal", "arm", "coords", "symbols")}
         for r in done if r.get("ok")], default=float) + "\n")
    tmp2.replace(OUT / f"{args.tag}_coords.json")
    print(f"[gxtb_series] {n_ok}/{len(tasks)} ok -> {OUT / (args.tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
