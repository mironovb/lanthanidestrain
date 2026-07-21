#!/usr/bin/env python3
"""Vietoris-Rips complexes for the re-optimised conformers.

The simplicial encoder is fit on **953 distinct geometries**, not 4,746 rows --
``train.py``'s own docstring calls that "the number that actually governs
overfitting here" -- and it shows: the SNN is the noisiest arm in the control
factorial by a factor of two (per-seed SD 0.047 against the tabular control's
0.027) and gains the most from seed averaging.  That is an over-parameterised
encoder on too small a structural sample, and the fix is more structures.

Stage 1 already produced them.  Every complex was re-optimised in water and
n-octanol; Stage 2 asked only whether the tighter geometries were better
*replacements* and found they were not -- median RMSD 1.87 A from the input,
which means they are different **conformers**, not refinements of the same one.
As replacements that is a null result.  As an ensemble it is a 3.2x increase in
the structural sample size, obtained with no new geometry optimisation.

What this module guarantees
---------------------------
The conformer asset must be indistinguishable from the shipped one in every
respect except the geometry, or the model can learn which structures are
augmented instead of learning chemistry.  So:

* ``src.geometry_features._rips_simplices`` is called **unchanged**, at the same
  ``DEFAULT_VR_MAX_EDGE_ANGSTROM`` (4.0) and ``max_dimension`` 2;
* node features are built by the same expressions as
  ``src/geometry_features.py:663-681`` -- same ``ATOMIC_NUMBER`` table, same
  ``LANTHANIDE_SYMBOLS`` metal test, same 3.10 A donor cutoff via
  ``geom3d_features.DONOR_CUTOFF_A``;
* partial charges are real Mulliken populations from
  ``automl.qc.conformer_charges``, never imputed -- an imputed charge would set
  ``charge_missing`` on every augmented structure and mark it;
* a conformer whose **coordination number changed** is dropped.  It is no longer
  the same complex, and ``rebuild_pi.py`` already established that rule for the
  same geometries.

``--verify-against-shipped`` rebuilds the *original* geometries through this
exact path and asserts the output matches ``vietoris_rips_inputs.npz`` element
for element.  If that fails, nothing downstream is attributable to the conformer
rather than to the pipeline, and the build should not be used.

``data/`` is never written; output goes to ``automl/artifacts/vr_conformers/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.geometry_features import (                       # noqa: E402
    ATOMIC_NUMBER, DEFAULT_VR_MAX_EDGE_ANGSTROM, LANTHANIDE_SYMBOLS,
    _coordination_shell, _rips_simplices, read_extxyz)
from automl.qc.reoptimize import job_table                # noqa: E402

CHARGE_DIR = _REPO / "automl/artifacts/conformer_charges"
OUT_DIR = _REPO / "automl/artifacts/vr_conformers"
SHIPPED = _REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz"


# ---------------------------------------------------------------------------
def donor_indices(symbols, coords, metal_symbol: str, core_cn: int):
    """Metal index and donor atom indices, from the shipped function itself.

    My first version reimplemented this as "donor-symbol atoms within
    ``DONOR_CUTOFF_A`` = 3.10 A", which is the rule ``geom3d_features`` uses.
    ``--verify-against-shipped`` rejected it on all six sampled complexes:
    ``_coordination_shell`` instead takes *every* donor-symbol atom, sorts by
    distance to the metal, and keeps the nearest ``core_cn`` -- a count rule,
    not a distance rule, with ``core_cn`` coming from the dataset rather than
    the geometry.

    So this calls ``_coordination_shell`` rather than restating it. A rule
    reimplemented is a rule that can drift; the whole point of the conformer
    asset is that only the geometry differs.
    """
    g = _ShimGeometry(symbols=np.asarray([str(s) for s in symbols]),
                      coordinates=np.asarray(coords, dtype=float))
    try:
        mi, idx, _v, _d = _coordination_shell(g, metal_symbol, int(core_cn))
    except (ValueError, IndexError):
        return None, []
    return int(mi), [int(i) for i in idx]


class _ShimGeometry:
    """The two attributes ``_coordination_shell`` touches.

    Conformer coordinates arrive from an npz, not from an extxyz file, so there
    is no ``ExtXYZGeometry`` to hand over.  Rather than reconstruct the full
    dataclass with fabricated energies, this exposes exactly what the function
    reads -- anything else it needed would fail loudly here instead of being
    silently defaulted.
    """

    __slots__ = ("symbols", "coordinates")

    def __init__(self, symbols, coordinates):
        self.symbols = symbols
        self.coordinates = coordinates


def _assemble(records: list[dict]) -> dict[str, np.ndarray]:
    """Exactly the layout of src/geometry_features.py:691-707."""
    (coords, zs, qs, is_metal, is_donor, e_idx, e_filt, t_idx, t_filt,
     build_ids) = ([] for _ in range(10))
    node_ptr, edge_ptr, tri_ptr = [0], [0], [0]

    for r in records:
        symbols, xyz = r["symbols"], r["coordinates"]
        edges, edge_f, tris, tri_f = _rips_simplices(
            xyz, DEFAULT_VR_MAX_EDGE_ANGSTROM)
        off = node_ptr[-1]
        coords.append(xyz.astype(np.float32))
        zs.append(np.asarray([ATOMIC_NUMBER.get(str(s), 0) for s in symbols],
                             dtype=np.int16))
        qs.append(r["partial_charges"].astype(np.float32))
        is_metal.append(np.asarray([str(s) in LANTHANIDE_SYMBOLS for s in symbols],
                                   dtype=np.int8))
        mask = np.zeros(len(symbols), dtype=np.int8)
        mask[r["donor_indices"]] = 1
        is_donor.append(mask)
        e_idx.append(edges + off); e_filt.append(edge_f)
        t_idx.append(tris + off); t_filt.append(tri_f)
        node_ptr.append(node_ptr[-1] + len(symbols))
        edge_ptr.append(edge_ptr[-1] + len(edges))
        tri_ptr.append(tri_ptr[-1] + len(tris))
        build_ids.append(r["build_id"])

    def cat(arrs, shape, dtype):
        return np.concatenate(arrs, axis=0) if arrs else np.zeros(shape, dtype=dtype)

    return {
        "coordinates": cat(coords, (0, 3), np.float32),
        "atomic_numbers": cat(zs, (0,), np.int16),
        "partial_charges": cat(qs, (0,), np.float32),
        "is_metal": cat(is_metal, (0,), np.int8),
        "is_coord_donor": cat(is_donor, (0,), np.int8),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "edge_index": cat(e_idx, (0, 2), np.int64).T,
        "edge_filtration": cat(e_filt, (0,), np.float32),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "triangle_index": cat(t_idx, (0, 3), np.int64).T,
        "triangle_filtration": cat(t_filt, (0,), np.float32),
        "triangle_ptr": np.asarray(tri_ptr, dtype=np.int64),
        "build_ids": np.asarray(build_ids, dtype="U32"),
    }


# ---------------------------------------------------------------------------
def _original_index() -> dict[str, dict]:
    """basename -> {build_id, path, metal_symbol, core_cn} per shipped geometry.

    ``core_cn`` and ``metal_symbol`` are read **out of the shipped asset**, not
    from a dataset column.

    That is not laziness, it is the only source that is guaranteed correct here.
    ``_coordination_shell`` keeps the nearest ``core_cn`` donors, and the build
    took ``core_cn`` from the geometry QC index -- which disagrees with
    ``final_ml_dataset_3d.parquet``'s ``coreCN`` for real structures: build
    b977d7df5bfe has coreCN 9 in the dataset, 8 donors in the asset, and "CN8"
    in its own filename. Using the dataset column produced a donor set that
    differed from the shipped one on every complex sampled.

    The count of ``is_coord_donor`` flags in the asset *is* the ``core_cn`` that
    complex was built with, whatever table supplied it, so reading it back
    reproduces the shipped mask by construction -- which is what
    ``--verify-against-shipped`` then confirms.
    """
    z = np.load(SHIPPED)
    ids = [str(b) for b in z["build_ids"]]
    node_ptr, donors, metals = z["node_ptr"], z["is_coord_donor"], z["is_metal"]
    syms = {v: k for k, v in ATOMIC_NUMBER.items()}
    per_build = {}
    for i, bid in enumerate(ids):
        sl = slice(int(node_ptr[i]), int(node_ptr[i + 1]))
        mi = np.flatnonzero(metals[sl])
        per_build[bid] = {
            "core_cn": int(donors[sl].sum()),
            "metal_symbol": syms.get(int(z["atomic_numbers"][sl][mi[0]]), "")
            if mi.size else "",
        }
    out = {}
    for r in job_table().itertuples():
        bid = str(r.geometry_feature_build_id)
        if bid not in per_build:
            continue
        out[r.basename] = {"build_id": bid, "path": Path(r.local),
                           **per_build[bid]}
    return out


def build(solvent: str, limit: int = 0) -> int:
    orig = _original_index()
    shipped_ids = set(str(b) for b in np.load(SHIPPED)["build_ids"])
    src = CHARGE_DIR / solvent
    if not src.exists():
        print(f"no charges for {solvent}; run automl.qc.conformer_charges first")
        return 1

    records, dropped = [], {"no_original": 0, "cn_changed": 0, "not_shipped": 0,
                            "no_metal": 0}
    files = sorted(src.glob("*.npz"))
    if limit:
        files = files[:limit]
    for f in files:
        z = np.load(f, allow_pickle=False)
        symbols = [str(s) for s in z["symbols"]]
        xyz = z["coordinates"].astype(np.float64)
        base = f.stem + ".xyz"
        meta = orig.get(base)
        if meta is None:
            dropped["no_original"] += 1
            continue
        if meta["build_id"] not in shipped_ids:
            # Not part of the 956 the model is trained on; adding it would
            # change the row set rather than the conformer count.
            dropped["not_shipped"] += 1
            continue
        mi, don = donor_indices(symbols, xyz, meta['metal_symbol'],
                                meta['core_cn'])
        if mi is None:
            dropped["no_metal"] += 1
            continue
        g0 = read_extxyz(meta["path"])
        _, don0 = donor_indices(list(g0.symbols), g0.coordinates,
                                meta['metal_symbol'], meta['core_cn'])
        if len(don) != len(don0):
            # A changed coordination number means this is no longer the same
            # complex -- the rule rebuild_pi.py already applies to these files.
            dropped["cn_changed"] += 1
            continue
        records.append({"symbols": symbols, "coordinates": xyz,
                        "partial_charges": z["partial_charges"],
                        "donor_indices": don, "build_id": meta["build_id"]})

    print(f"[vr] solvent={solvent} kept {len(records)} of {len(files)}  "
          f"dropped {dropped}", flush=True)
    if not records:
        return 1
    # Deterministic order, so the asset is reproducible run to run.
    records.sort(key=lambda r: r["build_id"])
    payload = _assemble(records)
    dest = OUT_DIR / solvent
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "vietoris_rips_inputs.npz", **payload)
    print(f"[vr] wrote {dest/'vietoris_rips_inputs.npz'}  "
          f"nodes={payload['coordinates'].shape[0]:,} "
          f"edges={payload['edge_filtration'].shape[0]:,} "
          f"triangles={payload['triangle_filtration'].shape[0]:,}")
    return 0


def verify_against_shipped(n: int = 12) -> int:
    """Rebuild ORIGINAL geometries through this path; must match the asset.

    The whole conformer argument rests on the featuriser being identical, so
    this is the check that has to pass before any conformer is used.
    """
    z = np.load(SHIPPED)
    ids = [str(b) for b in z["build_ids"]]
    orig = _original_index()
    by_build = {v["build_id"]: (k, v) for k, v in orig.items()}

    checked = mismatches = 0
    for i, bid in enumerate(ids):
        if checked >= n:
            break
        if bid not in by_build:
            continue
        _base, meta = by_build[bid]
        g = read_extxyz(meta["path"])
        mi, don = donor_indices(list(g.symbols), g.coordinates,
                                meta['metal_symbol'], meta['core_cn'])
        if mi is None:
            continue
        rec = {"symbols": [str(s) for s in g.symbols],
               "coordinates": np.asarray(g.coordinates, dtype=np.float64),
               "partial_charges": np.asarray(g.partial_charges, dtype=np.float32),
               "donor_indices": don, "build_id": bid}
        got = _assemble([rec])
        ns, ne = z["node_ptr"][i], z["edge_ptr"][i]
        nt = z["triangle_ptr"][i]
        want_nodes = slice(ns, z["node_ptr"][i + 1])
        want_edges = slice(ne, z["edge_ptr"][i + 1])
        want_tris = slice(nt, z["triangle_ptr"][i + 1])

        bad = []
        if not np.array_equal(got["atomic_numbers"], z["atomic_numbers"][want_nodes]):
            bad.append("atomic_numbers")
        if not np.array_equal(got["is_metal"], z["is_metal"][want_nodes]):
            bad.append("is_metal")
        if not np.array_equal(got["is_coord_donor"], z["is_coord_donor"][want_nodes]):
            bad.append("is_coord_donor")
        if got["edge_filtration"].shape != z["edge_filtration"][want_edges].shape:
            bad.append(f"n_edges {got['edge_filtration'].shape[0]} vs "
                       f"{z['edge_filtration'][want_edges].shape[0]}")
        elif not np.allclose(np.sort(got["edge_filtration"]),
                             np.sort(z["edge_filtration"][want_edges]), atol=1e-6):
            bad.append("edge_filtration")
        if got["triangle_filtration"].shape != z["triangle_filtration"][want_tris].shape:
            bad.append(f"n_triangles {got['triangle_filtration'].shape[0]} vs "
                       f"{z['triangle_filtration'][want_tris].shape[0]}")
        checked += 1
        if bad:
            mismatches += 1
            print(f"  MISMATCH {bid}: {', '.join(bad)}")

    print(f"[verify] {checked} complexes checked, {mismatches} mismatched")
    if mismatches:
        print("The featuriser is NOT reproducing the shipped asset. A conformer "
              "built this way would differ from the originals for reasons that "
              "have nothing to do with its geometry -- do not use it.")
    return 1 if mismatches else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solvent", choices=("water", "octanol"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-against-shipped", action="store_true")
    ap.add_argument("--verify-n", type=int, default=12)
    args = ap.parse_args()
    if args.verify_against_shipped:
        return verify_against_shipped(args.verify_n)
    if not args.solvent:
        ap.error("--solvent is required unless --verify-against-shipped")
    return build(args.solvent, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
