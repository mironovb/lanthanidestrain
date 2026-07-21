#!/usr/bin/env python3
"""Stage 2b: rebuild persistence images from the re-optimised geometries.

The topological result rests on persistence images computed from the *shipped*
geometries, which all stopped on a loose ``fmax = 0.2 eV/A`` criterion. The
re-optimised structures reach ~0.003 eV/A, and the adjacent-pair margin is
exactly where optimisation noise should hurt most: neighbouring lanthanides
differ by ~0.013 A in ionic radius while conformer/optimisation scatter in an
M-L distance is ~0.05 A.

This regenerates the images with **the same functions, constants and eligibility
rule** the shipped asset used -- `persistence_diagram` and `persistence_image`
from ``src/geometry_features.py``, resolution 20, spread 0.08, birth/death range
(0, 2.5), homology dims (0, 1). Nothing about the featurisation changes, so any
difference downstream is attributable to the geometry and not to the pipeline.

Output goes to ``automl/artifacts/pi_reopt/<solvent>/`` in the same npz schema as
the shipped asset (``images``, ``build_ids``), so ``PersistenceImages`` can load
it by path with no code change. ``data/`` is never written.
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

from src.geometry_features import (  # noqa: E402
    read_extxyz, persistence_diagram, persistence_image, PI_RESOLUTION)
from automl.qc.reoptimize import OUT_ROOT, job_table  # noqa: E402
from automl.qc.reopt_report import coordination  # noqa: E402

OUT_DIR = _REPO / "automl/artifacts/pi_reopt"


def build(solvent: str, limit: int = 0) -> int:
    src_dir = OUT_ROOT / solvent
    if not src_dir.exists():
        print(f"[rebuild-pi] no re-optimised geometries for {solvent!r}")
        return 1

    # geometry_key -> build_id, so the regenerated images key exactly as the
    # shipped ones do and every downstream join keeps working untouched.
    meta = pd.read_parquet(
        _REPO / "data/processed/final_ml_dataset_3d.parquet",
        columns=["geometry_key", "geometry_feature_build_id"]).dropna()
    key_to_build = dict(zip(meta["geometry_key"].astype(str),
                            meta["geometry_feature_build_id"].astype(str)))

    # Original geometries, to check coordination number is preserved.
    orig = job_table().set_index("basename")["local"].to_dict()

    images, build_ids, skipped = [], [], []
    cn_changed = 0
    files = sorted(src_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    for js in files:
        rec = json.loads(js.read_text())
        if not rec.get("ok"):
            skipped.append((rec.get("basename"), rec.get("reason", "failed")))
            continue
        # A structure whose coordination number changed is no longer the complex
        # the dataset row describes; featurising it would silently describe a
        # different molecule.
        bid = key_to_build.get(str(rec.get("geometry_key")))
        if bid is None:
            skipped.append((rec.get("basename"), "no_build_id"))
            continue
        try:
            g = read_extxyz(Path(rec["xyz"]))
            # Coordination-number audit, enforced rather than merely reported.
            # ~7% of re-optimisations change CN; those structures are no longer
            # the complex the dataset row describes, and featurising them would
            # silently substitute a different molecule.
            src = orig.get(str(rec.get("basename")))
            if src is not None:
                g0 = read_extxyz(Path(src))
                _, _, cn0 = coordination(g0.symbols, g0.coordinates)
                _, _, cn1 = coordination(g.symbols, g.coordinates)
                if cn0 != cn1:
                    cn_changed += 1
                    skipped.append((rec.get("basename"), f"cn_changed_{cn0}_to_{cn1}"))
                    continue
            img = persistence_image(persistence_diagram(g.coordinates),
                                    resolution=PI_RESOLUTION)
        except Exception as exc:
            skipped.append((rec.get("basename"), f"{type(exc).__name__}: {exc}"))
            continue
        images.append(img[np.newaxis, :, :])
        build_ids.append(bid)

    if not images:
        print("[rebuild-pi] nothing built")
        return 1
    arr = np.stack(images).astype(np.float32)
    out_dir = OUT_DIR / solvent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "complex_gfn2xtb_pi_images.npz"
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, images=arr,
                        build_ids=np.asarray(build_ids, dtype="U32"))
    tmp.replace(out)
    print(f"[rebuild-pi] {solvent}: {len(images)} images -> {out}")
    print(f"[rebuild-pi] shape={arr.shape}  finite={bool(np.isfinite(arr).all())}")
    print(f"[rebuild-pi] excluded for CN change: {cn_changed}")
    if skipped:
        print(f"[rebuild-pi] skipped {len(skipped)}:")
        for b, why in skipped[:6]:
            print(f"    {b}: {why}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvents", default="water,octanol")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rc = 0
    for s in [x for x in args.solvents.split(",") if x]:
        rc |= build(s, args.limit)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
