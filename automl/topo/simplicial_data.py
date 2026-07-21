#!/usr/bin/env python3
"""Simplicial complexes for message passing, from the shipped Vietoris-Rips asset.

``data/processed/feature_blocks/vietoris_rips_inputs.npz`` carries 956 complexes
as flat concatenated arrays with pointer offsets: 219,583 nodes, 2,301,232 edges
and 9,310,134 triangles, each edge/triangle tagged with its VR filtration radius
(max edge length 4.0 A).  Its manifest role is ``simplicial_model_input`` and it
had never been used by any experiment before this module.

What this file adds
-------------------
The asset gives the *boundary of edges* directly (``edge_index`` is a pair of
node ids) but not the *boundary of triangles* in terms of edges -- triangles are
stored as node triples.  Message passing needs edge ids, so the missing map

    triangle -> its three edge ids

is built here and cached.  Two properties of the asset make that cheap but are
not free assumptions, so both are asserted in the tests: node indices are
**global** (already offset by ``node_ptr``), and every simplex is stored with
ascending vertex order.  The edge list is *not* lexicographically sorted, so the
lookup builds its own sort order rather than calling ``searchsorted`` directly
on the raw array -- doing the latter silently returns wrong edge ids.

Sparsification
--------------
9,621 triangles per complex at the full 4.0 A filtration is dense and largely
redundant.  Two reductions are exposed as hyperparameters:

* ``filtration_max`` -- keep only simplices born below a radius.  Because the
  complex is a filtration, thresholding is exact: the result is the VR complex
  at the smaller radius, not an approximation.
* ``heavy_only`` -- drop hydrogens and re-index. Roughly halves the node count.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

VR_PATH = _REPO_ROOT / "data/processed/feature_blocks/vietoris_rips_inputs.npz"
CACHE_PATH = _REPO_ROOT / "automl/artifacts/topo/triangle_edges.npz"

# Elements present across the set; index 0 is reserved for padding/unknown so an
# nn.Embedding can be built directly on these ids.
_Z_VOCAB = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71)
Z_TO_IDX = {z: i + 1 for i, z in enumerate(_Z_VOCAB)}


def z_index(atomic_numbers: np.ndarray) -> np.ndarray:
    """Map atomic numbers to compact embedding ids (0 = unknown)."""
    out = np.zeros(len(atomic_numbers), dtype=np.int64)
    for z, i in Z_TO_IDX.items():
        out[atomic_numbers == z] = i
    return out


# ---------------------------------------------------------------------------
def build_triangle_edges(edge_index: np.ndarray, triangle_index: np.ndarray,
                         edge_ptr: np.ndarray, triangle_ptr: np.ndarray,
                         n_nodes_total: int, verbose: bool = True) -> np.ndarray:
    """Map every triangle to the ids of its three edges.

    Returns ``(3, n_triangles)`` int64 with **global** edge ids.  Raises if any
    triangle edge is absent from the edge list -- in a Vietoris-Rips complex a
    2-simplex cannot exist unless all three of its 1-faces do, so a miss means
    the asset is inconsistent and must not be silently patched over.
    """
    n_tri = triangle_index.shape[1]
    out = np.empty((3, n_tri), dtype=np.int64)
    n_complexes = len(edge_ptr) - 1
    for k in range(n_complexes):
        e0, e1 = int(edge_ptr[k]), int(edge_ptr[k + 1])
        t0, t1 = int(triangle_ptr[k]), int(triangle_ptr[k + 1])
        if t1 == t0:
            continue
        e = edge_index[:, e0:e1]
        # Encode an undirected edge (i<j) as a single integer key.
        key = e[0].astype(np.int64) * n_nodes_total + e[1].astype(np.int64)
        order = np.argsort(key, kind="stable")
        skey = key[order]
        t = triangle_index[:, t0:t1].astype(np.int64)
        for slot, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            q = t[a] * n_nodes_total + t[b]
            pos = np.searchsorted(skey, q)
            bad = (pos >= len(skey))
            pos = np.clip(pos, 0, len(skey) - 1)
            if bad.any() or not np.array_equal(skey[pos], q):
                n_missing = int(bad.sum() + (skey[pos] != q).sum())
                raise ValueError(
                    f"complex {k}: {n_missing} triangle edges absent from the "
                    f"edge list -- VR complex is inconsistent")
            out[slot, t0:t1] = order[pos] + e0
        if verbose and (k + 1) % 200 == 0:
            print(f"  [triangle_edges] {k + 1}/{n_complexes}", flush=True)
    return out


def load_triangle_edges(z: Any, cache: Path = CACHE_PATH,
                        verbose: bool = True) -> np.ndarray:
    """Cached ``build_triangle_edges`` (the build takes ~1 min, once)."""
    if cache.exists():
        with np.load(cache) as c:
            te = c["triangle_edges"]
        if te.shape[1] == z["triangle_index"].shape[1]:
            return te
    if verbose:
        print("[triangle_edges] building boundary map (cached afterwards)...",
              flush=True)
    te = build_triangle_edges(z["edge_index"], z["triangle_index"],
                              z["edge_ptr"], z["triangle_ptr"],
                              z["coordinates"].shape[0], verbose=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends ".npz" unless the name already ends in it, so
    # the temp file must itself end in .npz or the atomic rename below targets a
    # path that was never written.
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez_compressed(tmp, triangle_edges=te)
    tmp.replace(cache)
    return te


# ---------------------------------------------------------------------------
@dataclass
class Complex:
    """One simplicial complex, locally indexed from 0."""
    build_id: str
    z_idx: np.ndarray          # (N,) embedding ids
    coords: np.ndarray         # (N,3)
    charge: np.ndarray         # (N,)  non-finite values imputed to 0
    charge_missing: np.ndarray # (N,)  1 where the xTB charge was absent
    is_metal: np.ndarray       # (N,)
    is_donor: np.ndarray       # (N,)
    dist_to_metal: np.ndarray  # (N,)
    edge_index: np.ndarray     # (2,E) local node ids
    edge_filt: np.ndarray      # (E,)
    tri_edges: np.ndarray      # (3,T) local edge ids
    tri_filt: np.ndarray       # (T,)

    @property
    def n_nodes(self) -> int: return len(self.z_idx)
    @property
    def n_edges(self) -> int: return self.edge_index.shape[1]
    @property
    def n_tris(self) -> int: return self.tri_edges.shape[1]


class SimplicialComplexes:
    """Random access to the 956 complexes, with optional sparsification."""

    def __init__(self, vr_path: Path = VR_PATH, verbose: bool = True,
                 cache: Path | None = None):
        self.z = np.load(vr_path)
        self.build_ids = [str(b) for b in self.z["build_ids"]]
        # The boundary-map cache is keyed only by triangle count, so a second
        # asset loaded through this class would overwrite the shipped map -- and
        # if two assets ever happened to share a triangle count, it would return
        # the wrong one silently.  Conformer assets therefore pass their own path.
        self.triangle_edges = load_triangle_edges(
            self.z, cache=cache or CACHE_PATH, verbose=verbose)
        self.node_ptr = self.z["node_ptr"]
        self.edge_ptr = self.z["edge_ptr"]
        self.tri_ptr = self.z["triangle_ptr"]
        self._index = {b: i for i, b in enumerate(self.build_ids)}

    def __len__(self) -> int: return len(self.build_ids)

    def index_of(self, build_id: str) -> int | None:
        return self._index.get(str(build_id))

    def get(self, k: int, filtration_max: float | None = None,
            heavy_only: bool = False) -> Complex:
        z = self.z
        n0, n1 = int(self.node_ptr[k]), int(self.node_ptr[k + 1])
        e0, e1 = int(self.edge_ptr[k]), int(self.edge_ptr[k + 1])
        t0, t1 = int(self.tri_ptr[k]), int(self.tri_ptr[k + 1])

        zi = z["atomic_numbers"][n0:n1]
        coords = z["coordinates"][n0:n1].astype(np.float32)
        q = z["partial_charges"][n0:n1].astype(np.float32)
        # 292 of 219,583 shipped partial charges are non-finite (3 complexes,
        # the ones whose xTB properties are missing).  A single NaN node
        # feature does not stay local: message passing spreads it across the
        # complex and pooling turns the whole embedding into NaN, so those
        # complexes would emit NaN predictions for every row referencing them.
        # Impute to zero and carry an explicit missingness flag, so the model
        # can tell an imputed charge from a genuinely neutral atom.
        q_missing = (~np.isfinite(q)).astype(np.float32)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        metal = z["is_metal"][n0:n1].astype(np.float32)
        donor = z["is_coord_donor"][n0:n1].astype(np.float32)

        ei = z["edge_index"][:, e0:e1] - n0            # global -> local nodes
        ef = z["edge_filtration"][e0:e1]
        te = self.triangle_edges[:, t0:t1] - e0        # global -> local edges
        tf = z["triangle_filtration"][t0:t1]

        # --- filtration threshold (exact: yields the VR complex at that radius)
        if filtration_max is not None:
            ekeep = ef <= filtration_max
            tkeep = tf <= filtration_max
            emap = -np.ones(len(ef), dtype=np.int64)
            emap[ekeep] = np.arange(int(ekeep.sum()))
            ei, ef = ei[:, ekeep], ef[ekeep]
            te, tf = te[:, tkeep], tf[tkeep]
            te = emap[te]
            # a surviving triangle must keep all three faces; VR guarantees this
            # (a triangle's birth radius >= each of its edges'), assert it
            if te.size and te.min() < 0:
                raise AssertionError("triangle survived filtration but an edge "
                                     "did not -- filtration is not monotone")

        # --- heavy-atom restriction
        if heavy_only:
            keep = zi != 1
            nmap = -np.ones(len(zi), dtype=np.int64)
            nmap[keep] = np.arange(int(keep.sum()))
            zi, coords, q = zi[keep], coords[keep], q[keep]
            q_missing = q_missing[keep]
            metal, donor = metal[keep], donor[keep]
            ekeep = keep[ei[0]] & keep[ei[1]]
            emap = -np.ones(ei.shape[1], dtype=np.int64)
            emap[ekeep] = np.arange(int(ekeep.sum()))
            ei, ef = nmap[ei[:, ekeep]], ef[ekeep]
            tkeep = np.all(emap[te] >= 0, axis=0) if te.size else np.zeros(0, bool)
            te, tf = emap[te[:, tkeep]], tf[tkeep]

        mi = int(np.argmax(metal)) if metal.any() else 0
        d2m = np.linalg.norm(coords - coords[mi], axis=1).astype(np.float32)

        return Complex(build_id=self.build_ids[k], z_idx=z_index(zi),
                       coords=coords, charge=q, charge_missing=q_missing,
                       is_metal=metal, is_donor=donor,
                       dist_to_metal=d2m, edge_index=ei.astype(np.int64),
                       edge_filt=ef.astype(np.float32),
                       tri_edges=te.astype(np.int64),
                       tri_filt=tf.astype(np.float32))


# ---------------------------------------------------------------------------
def collate(complexes: list[Complex]) -> dict[str, Any]:
    """Batch complexes into flat tensors with per-level batch vectors.

    Offsets are applied so that node ids in ``edge_index`` and edge ids in
    ``tri_edges`` remain valid after concatenation.
    """
    import torch

    node_feats, eidx, tedg = [], [], []
    ef, tf = [], []
    nb, eb, tb, metal_rows = [], [], [], []
    n_off = e_off = 0
    for i, c in enumerate(complexes):
        node_feats.append(np.column_stack([
            c.charge, c.charge_missing, c.is_metal, c.is_donor,
            c.dist_to_metal]).astype(np.float32))
        eidx.append(c.edge_index + n_off)
        tedg.append(c.tri_edges + e_off)
        ef.append(c.edge_filt); tf.append(c.tri_filt)
        nb.append(np.full(c.n_nodes, i, dtype=np.int64))
        eb.append(np.full(c.n_edges, i, dtype=np.int64))
        tb.append(np.full(c.n_tris, i, dtype=np.int64))
        metal_rows.append(n_off + int(np.argmax(c.is_metal)))
        n_off += c.n_nodes
        e_off += c.n_edges

    t = torch.as_tensor
    return {
        "z_idx":      t(np.concatenate([c.z_idx for c in complexes])),
        "node_feat":  t(np.concatenate(node_feats)),
        "edge_index": t(np.concatenate(eidx, axis=1)),
        "edge_filt":  t(np.concatenate(ef)).unsqueeze(-1),
        "tri_edges":  t(np.concatenate(tedg, axis=1)),
        "tri_filt":   t(np.concatenate(tf)).unsqueeze(-1),
        "node_batch": t(np.concatenate(nb)),
        "edge_batch": t(np.concatenate(eb)),
        "tri_batch":  t(np.concatenate(tb)),
        "metal_index": t(np.asarray(metal_rows, dtype=np.int64)),
        "n_complexes": len(complexes),
    }


# ---------------------------------------------------------------------------
def selftest(verbose: bool = True) -> int:
    """Structural checks on the asset and the derived boundary map."""
    print("loading Vietoris-Rips asset...")
    S = SimplicialComplexes(verbose=verbose)
    print(f"complexes: {len(S)}")
    fails = 0

    rng = np.random.default_rng(0)
    picks = rng.choice(len(S), size=6, replace=False)

    for k in picks:
        c = S.get(int(k))
        # 1. triangle edges are exactly the three 1-faces of its vertex triple
        raw_t = S.z["triangle_index"][:, S.tri_ptr[k]:S.tri_ptr[k + 1]] - S.node_ptr[k]
        got = np.sort(c.edge_index[:, c.tri_edges.reshape(-1)]
                      .reshape(2, 3, -1), axis=0)
        want_pairs = np.stack([
            np.sort(raw_t[[0, 1]], axis=0),
            np.sort(raw_t[[0, 2]], axis=0),
            np.sort(raw_t[[1, 2]], axis=0)], axis=1)
        if not np.array_equal(got, want_pairs):
            print(f"  complex {k}: FAIL boundary map mismatch"); fails += 1
        # 2. every simplex has ascending vertex order
        if not (c.edge_index[0] < c.edge_index[1]).all():
            print(f"  complex {k}: FAIL edges not ascending"); fails += 1
        # 3. indices in range
        if c.edge_index.max() >= c.n_nodes or c.tri_edges.max() >= c.n_edges:
            print(f"  complex {k}: FAIL index out of range"); fails += 1
        # 4. exactly one metal
        if int(c.is_metal.sum()) != 1:
            print(f"  complex {k}: FAIL metal count {c.is_metal.sum()}"); fails += 1

    # 5. filtration thresholding is monotone and shrinks the complex
    c_full = S.get(int(picks[0]))
    c_cut = S.get(int(picks[0]), filtration_max=3.0)
    if not (c_cut.n_edges <= c_full.n_edges and c_cut.n_tris <= c_full.n_tris):
        print("  FAIL filtration threshold did not shrink the complex"); fails += 1

    # 6. heavy-atom restriction drops only hydrogens
    c_h = S.get(int(picks[0]), heavy_only=True)
    n_heavy = int((S.z["atomic_numbers"][S.node_ptr[picks[0]]:S.node_ptr[picks[0]+1]] != 1).sum())
    if c_h.n_nodes != n_heavy:
        print(f"  FAIL heavy_only node count {c_h.n_nodes} != {n_heavy}"); fails += 1

    # 7. batching a complex alone == batching it in a group (offset correctness)
    try:
        import torch
        a = collate([S.get(int(picks[0]))])
        g = collate([S.get(int(i)) for i in picks[:3]])
        n0 = a["z_idx"].numel()
        if not torch.equal(a["edge_index"], g["edge_index"][:, :a["edge_index"].shape[1]]):
            print("  FAIL batching changed edge_index for the first complex"); fails += 1
        if not torch.equal(a["z_idx"], g["z_idx"][:n0]):
            print("  FAIL batching changed node features"); fails += 1
    except ImportError:
        print("  (torch unavailable, batching check skipped)")

    print()
    print("SELFTEST PASSED" if fails == 0 else f"SELFTEST FAILED ({fails})")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
CONFORMER_ROOT = _REPO_ROOT / "automl/artifacts/vr_conformers"


class ConformerComplexes:
    """The shipped complexes plus their re-optimised conformers.

    Why this is a wrapper and not a change to ``SimplicialComplexes``: every run
    in the control factorial and everything published before it loads that class
    directly, and those runs have to stay reproducible bit for bit.  Nothing here
    modifies it.

    Conformer 0 is always the shipped geometry, so ``index_of`` and ``__len__``
    return exactly what ``SimplicialComplexes`` would.  That matters because
    ``build_row_table`` keys the row set off ``index_of``: if it changed, the
    conformer arm would be scored on a different set of rows than the arm it is
    being compared against, and the paired bootstrap would silently be comparing
    two datasets.

    Complexes differ in how many conformers survived -- a re-optimisation that
    failed, or that changed coordination number, is not available -- so
    ``n_conformers`` is per complex and callers must not assume a fixed count.
    """

    def __init__(self, vr_path: Path = VR_PATH, verbose: bool = False,
                 solvents: tuple[str, ...] = ("water", "octanol")):
        self.base = SimplicialComplexes(vr_path, verbose=verbose)
        self.build_ids = self.base.build_ids
        self._index = {b: i for i, b in enumerate(self.build_ids)}
        # conformer index k -> list of (asset, local index); slot 0 is the shipped
        self._alt: list[list[tuple[Any, int]]] = [[] for _ in self.build_ids]
        self.assets: list[Any] = []
        for solv in solvents:
            p = CONFORMER_ROOT / solv / "vietoris_rips_inputs.npz"
            if not p.exists():
                continue
            asset = SimplicialComplexes(
                p, verbose=verbose,
                cache=CONFORMER_ROOT / solv / "triangle_edges.npz")
            self.assets.append(asset)
            for j, bid in enumerate(asset.build_ids):
                k = self._index.get(str(bid))
                if k is not None:
                    self._alt[k].append((asset, j))

    def __len__(self) -> int:
        return len(self.build_ids)

    def index_of(self, build_id: str) -> int | None:
        return self._index.get(str(build_id))

    def n_conformers(self, k: int) -> int:
        return 1 + len(self._alt[k])

    def get(self, k: int, conformer: int = 0, filtration_max: float | None = None,
            heavy_only: bool = False) -> Complex:
        """``conformer`` 0 is the shipped geometry; 1.. are re-optimisations.

        Out-of-range indices wrap rather than raise: a caller sampling a random
        conformer per epoch should not have to special-case complexes with fewer
        of them, and wrapping degrades to "use the shipped geometry" instead of
        crashing a fold near the end of a long run.
        """
        if conformer <= 0 or not self._alt[k]:
            return self.base.get(k, filtration_max=filtration_max,
                                 heavy_only=heavy_only)
        asset, j = self._alt[k][(conformer - 1) % len(self._alt[k])]
        return asset.get(j, filtration_max=filtration_max, heavy_only=heavy_only)

    def coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for k in range(len(self)):
            counts[str(self.n_conformers(k))] = counts.get(
                str(self.n_conformers(k)), 0) + 1
        return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    S = SimplicialComplexes()
    if args.stats:
        for f in (None, 3.5, 3.0, 2.5):
            for heavy in (False, True):
                cs = [S.get(i, filtration_max=f, heavy_only=heavy)
                      for i in range(0, len(S), 100)]
                print(f"  filtration<={str(f):5s} heavy_only={heavy!s:5s} "
                      f"median nodes={int(np.median([c.n_nodes for c in cs])):4d} "
                      f"edges={int(np.median([c.n_edges for c in cs])):5d} "
                      f"tris={int(np.median([c.n_tris for c in cs])):6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
