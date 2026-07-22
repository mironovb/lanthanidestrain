#!/usr/bin/env python3
"""Cache the alpha-complex persistence diagrams once, so the sweep is affordable.

Why this exists
---------------
A persistence image is two independent things glued together: a *diagram*
(alpha-complex persistent homology of the coordinates) and a *rendering* of that
diagram onto a birth/death grid.  Only the rendering depends on the
hyperparameters we are sweeping -- resolution, Gaussian spread, birth/death
range, weighting, channel layout.  The diagram does not.

``src/geometry_features.persistence_diagram`` recomputes the alpha complex every
call, so a naive sweep would rebuild 953 alpha complexes per configuration.
Caching the diagrams once turns each configuration into a matrix product
(see ``pi_sweep_render.py``) and makes a ~50-configuration sweep cost about what
a single rebuild would.

What is cached, and why it is not just ``persistence_diagram``'s output
----------------------------------------------------------------------
``persistence_diagram`` discards the homology dimension, because the shipped
featurisation sums H0 and H1 into one image.  Whether that is the right thing to
do is one of the questions under test -- H0 deaths sit around 0.30 and H1 deaths
around 1.98, so they occupy essentially disjoint regions of the plane and are
being added together anyway.  The cache therefore keeps the dimension label.

To make sure keeping the label did not change anything else, ``_diagram_dims``
is checked against ``persistence_diagram`` itself on every complex under
``--strict``: filtering the cached points to ``PI_HOMOLOGY_DIMS`` must reproduce
the shipped function's output exactly.  Reuse-by-verification rather than
reuse-by-assumption; two coordination-rule bugs earlier in this study were caught
by exactly this kind of check.

The coordinate source, and what the gate found
----------------------------------------------
The shipped asset was built from the full ``read_extxyz`` coordinates (all atoms,
hydrogens included) of the geometries ``automl.qc.reoptimize.job_table`` resolves
on disk -- *not* from ``vietoris_rips_inputs.npz``, which is heavy-atom only.

Rendering the cache at the shipped settings reproduces
``complex_gfn2xtb_pi_images.npz`` exactly for **935 of the 953** complexes.  The
remaining 18 do not match, and the gate establishes that this is not a defect
here: calling the shipped ``persistence_diagram`` / ``persistence_image``
*directly* on the same coordinates reproduces each mismatch exactly and
deterministically under gudhi 3.13.0.  All 18 are large (268-361 atoms), which is
where alpha-complex construction is most sensitive to the CGAL geometric
predicates.  The shipped asset was built under a different gudhi/CGAL and cannot
be reproduced bit-for-bit in this environment by *any* code path, including its
own.

The consequence is handled rather than ignored: ``pi_sweep_render`` sweeps the
shipped configuration as an explicit **reproduction anchor**, rendered from this
same cache.  Every "tuned versus untuned" comparison is therefore between two
image sets built the same way in the same environment, instead of one against a
differently-built asset.  See ``shipped_reproduction.json``.

``--verify-against-shipped`` is a gate on *attribution*: a complex that disagrees
with this renderer but agrees with the shipped functions is a bug here, and
fails.

``data/`` is never written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.geometry_features import (  # noqa: E402
    read_extxyz, persistence_diagram, PI_HOMOLOGY_DIMS)
from automl.qc.reoptimize import job_table  # noqa: E402

SHIPPED = _REPO / "data/processed/feature_blocks/complex_gfn2xtb_pi_images.npz"
OUT_DIR = _REPO / "automl/artifacts/pi_sweep"
CACHE = OUT_DIR / "diagrams.npz"


def _diagram_dims(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Persistence points with their homology dimension retained.

    Mirrors ``persistence_diagram`` exactly -- same alpha complex, same
    finiteness and ``death > birth`` filter, same iteration order -- but keeps
    every dimension rather than only ``PI_HOMOLOGY_DIMS``, and returns the label
    alongside.
    """
    import gudhi as gd
    if coordinates.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int8)
    tree = gd.AlphaComplex(points=coordinates).create_simplex_tree()
    pts: list[tuple[float, float]] = []
    dims: list[int] = []
    for dim, (birth, death) in tree.persistence():
        if np.isfinite(birth) and np.isfinite(death) and death > birth:
            pts.append((float(birth), float(death)))
            dims.append(int(dim))
    if not pts:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int8)
    return (np.asarray(pts, dtype=np.float64),
            np.asarray(dims, dtype=np.int8))


def shipped_build_ids() -> list[str]:
    with np.load(SHIPPED) as z:
        return [str(b) for b in z["build_ids"]]


def geometry_paths() -> dict[str, str]:
    """build_id -> local .xyz, first occurrence wins (as ``rebuild_pi`` does)."""
    jt = job_table()
    out: dict[str, str] = {}
    for bid, loc in zip(jt["geometry_feature_build_id"].astype(str), jt["local"]):
        out.setdefault(bid, loc)
    return out


def build(strict: bool = True, limit: int = 0) -> int:
    ids = shipped_build_ids()
    paths = geometry_paths()
    missing = [b for b in ids if b not in paths]
    if missing:
        raise SystemExit(f"{len(missing)} shipped build_ids have no local .xyz; "
                         f"first: {missing[:3]}")
    if limit:
        ids = ids[:limit]

    pts_all: list[np.ndarray] = []
    dim_all: list[np.ndarray] = []
    ptr = [0]
    n_atoms = []
    for i, bid in enumerate(ids):
        g = read_extxyz(Path(paths[bid]))
        p, d = _diagram_dims(g.coordinates)
        if strict:
            # Filtering the cache to the shipped dimensions must reproduce the
            # shipped function bit for bit, or the cache is not the same object.
            ref = persistence_diagram(g.coordinates)
            mine = p[np.isin(d, PI_HOMOLOGY_DIMS)]
            if mine.shape != ref.shape or not np.array_equal(mine, ref):
                raise SystemExit(
                    f"{bid}: cached diagram disagrees with persistence_diagram "
                    f"({mine.shape} vs {ref.shape})")
        pts_all.append(p)
        dim_all.append(d)
        ptr.append(ptr[-1] + len(p))
        n_atoms.append(len(g.symbols))
        if (i + 1) % 100 == 0:
            print(f"[pi-build] {i+1}/{len(ids)} complexes, "
                  f"{ptr[-1]} points", flush=True)

    points = (np.concatenate(pts_all) if pts_all
              else np.zeros((0, 2), dtype=np.float64))
    dims = (np.concatenate(dim_all) if dim_all
            else np.zeros(0, dtype=np.int8))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_name(CACHE.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, points=points, dims=dims,
        ptr=np.asarray(ptr, dtype=np.int64),
        build_ids=np.asarray(ids, dtype="U32"),
        n_atoms=np.asarray(n_atoms, dtype=np.int32))
    tmp.replace(CACHE)

    per = np.diff(ptr)
    print(f"[pi-build] {len(ids)} complexes, {len(points)} points -> {CACHE}")
    print(f"[pi-build] points/complex: median {np.median(per):.0f}  "
          f"min {per.min()}  max {per.max()}")
    for dim in (0, 1, 2, 3):
        k = int((dims == dim).sum())
        if k:
            sub = points[dims == dim]
            print(f"[pi-build]   H{dim}: {k:7d} points  "
                  f"death median {np.median(sub[:, 1]):.3f}  "
                  f"p95 {np.percentile(sub[:, 1], 95):.3f}  "
                  f"max {sub[:, 1].max():.3f}")
    return 0


def load() -> dict:
    if not CACHE.exists():
        raise SystemExit(f"no diagram cache at {CACHE}; run --build first")
    with np.load(CACHE) as z:
        return {"points": z["points"], "dims": z["dims"], "ptr": z["ptr"],
                "build_ids": [str(b) for b in z["build_ids"]]}


def verify_against_shipped(tol: float = 1e-5) -> int:
    """Render the cache at the shipped settings and compare to the asset.

    The gate.  What it has to establish is that the cache *is* the object the
    published persistence-image arm was built from, so that a tuned
    configuration can be compared against it.

    On first run this found that 18 of the 953 complexes do not match -- and
    that the discrepancy is **not** in this cache.  Calling the shipped
    ``persistence_diagram`` / ``persistence_image`` directly on the same
    coordinates reproduces the mismatch exactly and deterministically, on gudhi
    3.13.0.  Every affected complex is large (268-361 atoms), which is where
    alpha-complex construction is most sensitive to the CGAL predicates: the
    shipped asset was built under a different gudhi/CGAL and cannot be
    reproduced bit-for-bit in this environment by any code path, including its
    own.

    So the gate tests the attribution rather than asserting it.  For every
    mismatching complex it re-runs the *shipped functions* and requires them to
    mismatch by the same amount.  If a complex disagrees with this renderer but
    agrees with the shipped functions, that is a bug here and the gate fails.

    The consequence for the sweep is handled in ``pi_sweep_render``: the shipped
    configuration is rendered from this same cache and swept as an explicit
    anchor, so "tuned versus untuned" is a comparison between two images built
    the same way, rather than one against a differently-built asset.
    """
    from automl.qc.pi_sweep_render import SHIPPED_CONFIG, render_all
    from src.geometry_features import (read_extxyz, persistence_diagram,
                                       persistence_image, PI_RESOLUTION)

    cache = load()
    imgs, ids = render_all(cache, SHIPPED_CONFIG)
    with np.load(SHIPPED) as z:
        ref = z["images"].astype(np.float32)
        ref_ids = [str(b) for b in z["build_ids"]]

    if ids != ref_ids:
        print(f"[pi-gate] build_id order differs "
              f"({len(ids)} vs {len(ref_ids)}) -- realigning")
        pos = {b: i for i, b in enumerate(ids)}
        if not set(ref_ids) <= set(pos):
            print("[pi-gate] FAIL: shipped ids are not a subset of the cache")
            return 1
        imgs = imgs[[pos[b] for b in ref_ids]]

    if imgs.shape != ref.shape:
        print(f"[pi-gate] FAIL: shape {imgs.shape} vs shipped {ref.shape}")
        return 1

    n = len(ref_ids)
    flat_dev = np.abs(imgs - ref).reshape(n, -1).max(axis=1)
    scale = np.maximum(np.abs(ref).reshape(n, -1).max(axis=1), 1e-12)
    rel = flat_dev / scale
    bad = np.where(rel > tol)[0]
    print(f"[pi-gate] rendered {imgs.shape} at shipped settings")
    print(f"[pi-gate] exact matches: {n - len(bad)}/{n} "
          f"({100*(n-len(bad))/n:.1f} %)")
    if len(bad) == 0:
        print("[pi-gate] PASS -- the cache reproduces "
              "complex_gfn2xtb_pi_images.npz exactly")
        return 0

    # Attribute each mismatch: does the shipped code path reproduce the asset
    # where this renderer does not?  If so, the bug is here.
    paths = geometry_paths()
    mine_only = []
    for i in bad:
        bid = ref_ids[i]
        g = read_extxyz(Path(paths[bid]))
        direct = persistence_image(persistence_diagram(g.coordinates),
                                   resolution=PI_RESOLUTION)
        r_direct = float(np.abs(direct - ref[i, 0]).max()) / scale[i]
        if r_direct <= tol:
            mine_only.append((bid, float(rel[i])))
        elif abs(r_direct - rel[i]) > 1e-6:
            mine_only.append((bid, float(rel[i])))

    if mine_only:
        print(f"[pi-gate] FAIL -- {len(mine_only)} complexes disagree with this "
              f"renderer but not with the shipped functions; that is a bug here")
        for bid, r in mine_only[:5]:
            print(f"[pi-gate]     {bid}  relative {r:.3e}")
        return 1

    note = OUT_DIR / "shipped_reproduction.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    note.write_text(json.dumps({
        "n_complexes": n,
        "n_exact": int(n - len(bad)),
        "n_environment_mismatch": int(len(bad)),
        "max_relative_deviation": float(rel.max()),
        "attribution": "the shipped persistence_diagram/persistence_image, "
                       "called directly on the same coordinates, reproduce "
                       "these mismatches exactly and deterministically; the "
                       "shipped asset was built under a different gudhi/CGAL",
        "gudhi": _gudhi_version(),
        "build_ids": [ref_ids[i] for i in bad],
    }, indent=2) + "\n")
    print(f"[pi-gate] {len(bad)} complexes differ, and the shipped functions "
          f"differ identically -- attributable to the gudhi/CGAL version, "
          f"not to this cache")
    print(f"[pi-gate] recorded -> {note.name}")
    print("[pi-gate] PASS (with a documented environment discrepancy)")
    return 0


def _gudhi_version() -> str:
    try:
        import gudhi
        return str(gudhi.__version__)
    except Exception:                       # pragma: no cover
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="skip the per-complex agreement check with "
                         "persistence_diagram (roughly halves build time)")
    ap.add_argument("--verify-against-shipped", action="store_true")
    args = ap.parse_args()
    rc = 0
    if args.build:
        rc |= build(strict=args.strict, limit=args.limit)
    if args.verify_against_shipped:
        rc |= verify_against_shipped()
    if not (args.build or args.verify_against_shipped):
        ap.print_help()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
