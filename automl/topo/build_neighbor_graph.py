"""Neighbour graphs beyond the shipped 4.0 A ceiling.

Why this exists now rather than earlier.  Every published run thresholded the
Vietoris-Rips asset at 3.5 A, and CAMPAIGN6 screening found that simply using
the whole shipped asset -- ``--filtration-max 4.0`` -- is worth **+0.0419**, and
+0.0711 once the radial basis is widened to match.  The receptive field is
therefore a live axis, and 4.0 A is not a modelling choice: it is the largest
edge the shipped file contains.  This module removes that ceiling by
recomputing edges from the ``coordinates`` array the asset already carries.

The two prior wide-field runs (``snn_filt5`` -0.0686, ``snn_allatom`` -0.3273)
do not bound this.  Both have ``pair_loss_weight = None, select_on = None`` in
their recorded configs -- they predate the contrast objective entirely.

``data/`` is read-only.  This reads one file from it and writes only under
``automl/artifacts/vr_cutoff/``.

    python3 -m automl.topo.build_neighbor_graph --verify-against-shipped
    python3 -m automl.topo.build_neighbor_graph --name c50 --cutoff 5.0
    python3 -m automl.topo.build_neighbor_graph --name k24 --knn 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz"
OUT_ROOT = REPO / "automl/artifacts/vr_cutoff"


def _edges_cutoff(xyz: np.ndarray, cutoff: float):
    """All pairs within ``cutoff``, ascending vertex order, with distances.

    This IS the 1-skeleton of the Rips complex: a VR edge is "the two points
    are within max_edge" and its filtration value is their distance.  So no
    gudhi dependency and no reimplementation risk -- and
    ``--verify-against-shipped`` proves the claim rather than asserting it.
    """
    from scipy.spatial import cKDTree
    pairs = cKDTree(xyz).query_pairs(float(cutoff), output_type="ndarray")
    if not len(pairs):
        return np.zeros((0, 2), np.int64), np.zeros(0, np.float32)
    pairs = np.sort(pairs, axis=1)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order].astype(np.int64)
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    return pairs, d.astype(np.float32)


def _edges_knn(xyz: np.ndarray, k: int):
    """Symmetrised k-nearest-neighbour graph.

    A DEGREE rule rather than a distance rule.  At a fixed radius a dense
    region gives a node 200 neighbours while a sparse one gives 8, and the
    message-passing normalisation (scatter_mean) then means different things in
    the two -- one live reading of why simply widening the radius might stop
    helping.  k-NN holds degree fixed and lets the radius vary instead.
    """
    from scipy.spatial import cKDTree
    n = len(xyz)
    kk = min(int(k) + 1, n)
    _d, idx = cKDTree(xyz).query(xyz, k=kk)
    src = np.repeat(np.arange(n), kk - 1)
    dst = idx[:, 1:].reshape(-1)
    pairs = np.sort(np.column_stack([src, dst]), axis=1)
    pairs = np.unique(pairs, axis=0).astype(np.int64)
    d = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    return pairs, d.astype(np.float32)


def build(name: str, cutoff: float | None = None, knn: int | None = None,
          verbose: bool = True) -> Path:
    if (cutoff is None) == (knn is None):
        raise SystemExit("give exactly one of --cutoff / --knn")
    z = np.load(SHIPPED)
    node_ptr = z["node_ptr"]
    co = z["coordinates"]
    n_cplx = len(node_ptr) - 1

    e_idx, e_filt, e_ptr = [], [], [0]
    t0 = time.time()
    for i in range(n_cplx):
        a, b = int(node_ptr[i]), int(node_ptr[i + 1])
        p, d = (_edges_cutoff(co[a:b], cutoff) if cutoff is not None
                else _edges_knn(co[a:b], knn))
        e_idx.append(p + a)                       # global node indexing
        e_filt.append(d)
        e_ptr.append(e_ptr[-1] + len(p))
    edge_index = (np.concatenate(e_idx).T.astype(np.int64) if e_idx
                  else np.zeros((2, 0), np.int64))
    edge_filtration = np.concatenate(e_filt).astype(np.float32)
    edge_ptr = np.asarray(e_ptr, dtype=np.int64)

    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(
        coordinates=co, atomic_numbers=z["atomic_numbers"],
        partial_charges=z["partial_charges"], is_metal=z["is_metal"],
        is_coord_donor=z["is_coord_donor"], node_ptr=node_ptr,
        edge_index=edge_index, edge_filtration=edge_filtration,
        edge_ptr=edge_ptr,
        # NO triangles.  They scale roughly as r^6 -- the shipped 4.0 A asset
        # already holds 9.3M, so 6.0 A would be ~10^8, several GB before the
        # triangle->edge cache is built, and it would confound "wider field"
        # with "10x more 2-simplices" in one cell.  train.py hard-gates these
        # assets to --arch dist / --no-triangles for exactly this reason.
        triangle_index=np.zeros((3, 0), np.int64),
        triangle_filtration=np.zeros(0, np.float32),
        triangle_ptr=np.zeros(n_cplx + 1, np.int64),
        # Same build ids in the SAME ORDER as shipped: build_row_table resolves
        # rows through this list, so reordering would silently repoint every row
        # at a different complex.
        build_ids=z["build_ids"],
    )
    # NOTE the tmp name must itself end in .npz: savez_compressed APPENDS
    # ".npz" when the given name lacks it, so "...npz.tmp" would silently be
    # written as "...npz.tmp.npz" and the atomic replace would then fail on a
    # path that was never created.
    tmp = out / "vietoris_rips_inputs.tmp.npz"
    np.savez_compressed(tmp, **payload)
    tmp.replace(out / "vietoris_rips_inputs.npz")

    deg = 2.0 * edge_index.shape[1] / max(len(co), 1)
    meta = {"name": name, "cutoff": cutoff, "knn": knn,
            "n_complexes": n_cplx, "n_nodes": int(len(co)),
            "n_edges": int(edge_index.shape[1]),
            "mean_degree": round(deg, 2),
            "max_edge_len": float(edge_filtration.max()) if len(edge_filtration) else 0.0,
            "triangles": 0,
            "source": str(SHIPPED.relative_to(REPO)),
            "source_sha256": hashlib.sha256(
                SHIPPED.read_bytes()).hexdigest()[:16],
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if verbose:
        print(f"[nbr] {name}: {meta['n_edges']:,} edges, mean degree "
              f"{deg:.1f}, max {meta['max_edge_len']:.2f} A "
              f"[{meta['seconds']:.0f}s] -> {out}")
    return out


def verify_against_shipped() -> int:
    """Rebuild at 4.0 A and require the shipped edge set back, exactly.

    The do-no-harm gate.  If the reconstruction of a cutoff the shipped file
    already contains does not reproduce it, no larger cutoff from this code can
    be trusted either.
    """
    z = np.load(SHIPPED)
    node_ptr, co = z["node_ptr"], z["coordinates"]
    ship_idx, ship_ptr = z["edge_index"], z["edge_ptr"]
    bad = 0
    for i in range(len(node_ptr) - 1):
        a, b = int(node_ptr[i]), int(node_ptr[i + 1])
        p, _d = _edges_cutoff(co[a:b], 4.0)
        got = {(int(u), int(v)) for u, v in p + a}
        lo, hi = int(ship_ptr[i]), int(ship_ptr[i + 1])
        want = {(int(u), int(v)) for u, v in ship_idx[:, lo:hi].T}
        want = {(min(u, v), max(u, v)) for u, v in want}
        bad += len(got ^ want)
    total = int(ship_idx.shape[1])
    frac = bad / max(total, 1)
    print(f"[nbr] verify vs shipped 4.0 A: {bad} edge disagreements of "
          f"{total:,} ({frac:.4%})")
    # float32 coordinates put a handful of pairs exactly on the boundary; a
    # tiny disagreement is arithmetic, a large one is a bug.
    if frac > 1e-4:
        print("[nbr] FAIL: reconstruction does not reproduce the shipped set")
        return 1
    print("[nbr] OK -- the 1-skeleton reconstruction is faithful")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name")
    ap.add_argument("--cutoff", type=float, default=None)
    ap.add_argument("--knn", type=int, default=None)
    ap.add_argument("--verify-against-shipped", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="build the standard set: c50, c60, c80, k24")
    args = ap.parse_args()

    if args.verify_against_shipped:
        return verify_against_shipped()
    if args.all:
        rc = verify_against_shipped()
        if rc:
            return rc
        for nm, c, k in (("c50", 5.0, None), ("c60", 6.0, None),
                         ("c80", 8.0, None), ("k24", None, 24)):
            build(nm, cutoff=c, knn=k)
        return 0
    if not args.name:
        raise SystemExit("give --name with --cutoff/--knn, or --all")
    build(args.name, cutoff=args.cutoff, knn=args.knn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
