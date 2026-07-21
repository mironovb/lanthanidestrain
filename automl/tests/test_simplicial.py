#!/usr/bin/env python3
"""Correctness tests for the simplicial data pipeline and the MPSN.

These are the checks that make an end-to-end topological model trustworthy. A
message-passing network that silently breaks permutation or rotation invariance
still trains and still reports a number -- it just reports a wrong one -- so the
invariances are asserted rather than assumed.

Run:  python3 -m pytest automl/tests/test_simplicial.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from automl.topo.simplicial_data import (  # noqa: E402
    SimplicialComplexes, collate, z_index, Z_TO_IDX)
from automl.topo.snn import (  # noqa: E402
    SimplicialNet, scatter_mean, scatter_max, scatter_sum, count_parameters)


@pytest.fixture(scope="module")
def complexes():
    return SimplicialComplexes(verbose=False)


# ---------------------------------------------------------------------------
# Scatter primitives
# ---------------------------------------------------------------------------
def test_scatter_sum_matches_manual():
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx = torch.tensor([0, 1, 0])
    out = scatter_sum(src, idx, 2)
    assert torch.allclose(out, torch.tensor([[6.0, 8.0], [3.0, 4.0]]))


def test_scatter_mean_matches_manual():
    src = torch.tensor([[1.0], [3.0], [5.0]])
    idx = torch.tensor([0, 0, 1])
    assert torch.allclose(scatter_mean(src, idx, 2), torch.tensor([[2.0], [5.0]]))


def test_scatter_max_handles_empty_groups():
    src = torch.tensor([[1.0], [7.0]])
    idx = torch.tensor([0, 0])
    out = scatter_max(src, idx, 3)          # group 1 and 2 are empty
    assert out[0].item() == 7.0
    assert torch.isfinite(out).all(), "empty groups must not leak -inf"


# ---------------------------------------------------------------------------
# Boundary structure
# ---------------------------------------------------------------------------
def test_triangle_edges_are_the_three_faces(complexes):
    """Each triangle's edge ids must span exactly its three vertex pairs."""
    S = complexes
    for k in (0, 17, 123, 500):
        c = S.get(k)
        if c.n_tris == 0:
            continue
        raw = S.z["triangle_index"][:, S.tri_ptr[k]:S.tri_ptr[k + 1]] - S.node_ptr[k]
        got = np.sort(c.edge_index[:, c.tri_edges.reshape(-1)].reshape(2, 3, -1),
                      axis=0)
        want = np.stack([np.sort(raw[[0, 1]], axis=0),
                         np.sort(raw[[0, 2]], axis=0),
                         np.sort(raw[[1, 2]], axis=0)], axis=1)
        assert np.array_equal(got, want), f"complex {k}: boundary map wrong"


def test_boundary_composition_is_zero_mod_2(complexes):
    """d1 . d2 = 0 over GF(2): every vertex of a triangle appears twice."""
    S = complexes
    for k in (3, 88, 400):
        c = S.get(k)
        if c.n_tris == 0:
            continue
        verts = c.edge_index[:, c.tri_edges.reshape(-1)].reshape(2, 3, -1)
        for t in range(min(c.n_tris, 400)):
            flat = verts[:, :, t].reshape(-1)
            counts = np.bincount(flat)
            nz = counts[counts > 0]
            assert (nz % 2 == 0).all(), "d1.d2 != 0 mod 2"


def test_filtration_threshold_is_a_subcomplex(complexes):
    """Thresholding must give a genuine subcomplex, not a truncation."""
    S = complexes
    c_full = S.get(7)
    c_cut = S.get(7, filtration_max=3.0)
    assert c_cut.n_edges <= c_full.n_edges
    assert c_cut.n_tris <= c_full.n_tris
    assert (c_cut.edge_filt <= 3.0 + 1e-6).all()
    assert (c_cut.tri_filt <= 3.0 + 1e-6).all()
    if c_cut.n_tris:
        assert c_cut.tri_edges.max() < c_cut.n_edges, "dangling edge reference"


def test_heavy_only_drops_exactly_hydrogens(complexes):
    S = complexes
    k = 11
    c = S.get(k, heavy_only=True)
    zi = S.z["atomic_numbers"][S.node_ptr[k]:S.node_ptr[k + 1]]
    assert c.n_nodes == int((zi != 1).sum())
    assert c.z_idx.min() >= 1
    assert Z_TO_IDX[1] not in set(c.z_idx.tolist()), "hydrogen survived"


def test_exactly_one_metal_per_complex(complexes):
    S = complexes
    for k in (0, 50, 300, 900):
        assert int(S.get(k).is_metal.sum()) == 1


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def test_batching_preserves_single_complex(complexes):
    S = complexes
    alone = collate([S.get(5)])
    grouped = collate([S.get(5), S.get(6), S.get(7)])
    n = alone["z_idx"].numel()
    e = alone["edge_index"].shape[1]
    assert torch.equal(alone["z_idx"], grouped["z_idx"][:n])
    assert torch.equal(alone["edge_index"], grouped["edge_index"][:, :e])
    assert alone["metal_index"][0] == grouped["metal_index"][0]


def test_batch_offsets_stay_in_range(complexes):
    S = complexes
    b = collate([S.get(i) for i in (1, 2, 3, 4)])
    assert int(b["edge_index"].max()) < b["z_idx"].numel()
    assert int(b["tri_edges"].max()) < b["edge_index"].shape[1]
    assert int(b["metal_index"].max()) < b["z_idx"].numel()


# ---------------------------------------------------------------------------
# Model invariances -- the ones that matter
# ---------------------------------------------------------------------------
def _model(tabular_dim=0, seed=0):
    torch.manual_seed(seed)
    m = SimplicialNet(dim=16, layers=2, dropout=0.0, tabular_dim=tabular_dim)
    m.eval()
    return m


def test_output_is_invariant_to_rigid_motion(complexes):
    """Rotating and translating the molecule must not change the prediction.

    Invariance here is structural -- no raw coordinate is fed to the network --
    so this test guards against a future edit that adds one.
    """
    S = complexes
    c = S.get(9, heavy_only=True)
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rot = c.coords @ q.T + np.array([3.0, -1.5, 0.7], dtype=np.float32)

    import copy
    c2 = copy.deepcopy(c)
    c2.coords = rot.astype(np.float32)
    # distance-to-metal is the only geometric input and is rigid-invariant
    mi = int(np.argmax(c2.is_metal))
    c2.dist_to_metal = np.linalg.norm(c2.coords - c2.coords[mi],
                                      axis=1).astype(np.float32)

    m = _model()
    with torch.no_grad():
        a = m(collate([c]))
        b = m(collate([c2]))
    assert torch.allclose(a, b, atol=1e-5), "model is not rigid-motion invariant"


def test_output_is_invariant_to_node_permutation(complexes):
    """Relabelling atoms must not change the prediction."""
    S = complexes
    c = S.get(9, heavy_only=True)
    rng = np.random.default_rng(1)
    perm = rng.permutation(c.n_nodes)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(c.n_nodes)

    import copy
    c2 = copy.deepcopy(c)
    for attr in ("z_idx", "coords", "charge", "charge_missing", "is_metal",
                 "is_donor", "dist_to_metal"):
        setattr(c2, attr, getattr(c, attr)[perm])
    remapped = inv[c.edge_index]
    c2.edge_index = np.sort(remapped, axis=0)   # keep the i<j convention

    m = _model()
    with torch.no_grad():
        a = m(collate([c]))
        b = m(collate([c2]))
    assert torch.allclose(a, b, atol=1e-5), "model is not permutation invariant"


def test_batch_independence(complexes):
    """A complex's prediction must not depend on what it is batched with."""
    S = complexes
    cs = [S.get(i, heavy_only=True) for i in (20, 21, 22)]
    m = _model()
    with torch.no_grad():
        alone = m(collate([cs[0]]))
        together = m(collate(cs))
    assert torch.allclose(alone, together[:1], atol=1e-5), \
        "prediction leaks across the batch"


def test_hybrid_head_consumes_tabular_features(complexes):
    S = complexes
    c = S.get(30, heavy_only=True)
    m = _model(tabular_dim=8)
    tab = torch.zeros(1, 8)
    with torch.no_grad():
        a = m(collate([c]), tabular=tab)
        b = m(collate([c]), tabular=tab + 1.0)
    assert not torch.allclose(a, b), "tabular block is being ignored"


def test_model_is_small_enough_for_the_data(complexes):
    """953 distinct structures cannot support an arbitrarily large encoder."""
    m = SimplicialNet(dim=96, layers=3, dropout=0.1, tabular_dim=64)
    n = count_parameters(m)
    assert n < 3_000_000, f"encoder has {n} parameters for 953 structures"


# ---------------------------------------------------------------------------
# Probe leakage guard
# ---------------------------------------------------------------------------
def test_z_vocab_makes_the_metal_element_readable(complexes):
    """Documents *why* the metal probe has to mask the element.

    Every lanthanide 57-71 gets its own embedding token, so a model asked to
    predict the lanthanide index can read it straight off the metal node
    without using any geometry.  Measured that way the probe scores
    R2 = 0.9995 -- a readout of the label, not a statement about structure.
    If this ever stops being true the masking can be revisited; until then it
    is mandatory.
    """
    from automl.topo.simplicial_data import Z_TO_IDX
    lanthanide_tokens = {Z_TO_IDX[z] for z in range(57, 72) if z in Z_TO_IDX}
    assert len(lanthanide_tokens) > 1, \
        "lanthanides share a token; the probe masking may no longer be needed"


def test_masked_metal_cache_hides_the_element(complexes):
    """The masked probe must remove element identity and nothing else."""
    from automl.topo.metal_probe import MaskedMetalCache, GENERIC_LN

    class _Inner:
        def __init__(self, S): self.S = S
        def batch(self, ids): return collate([self.S.get(i, heavy_only=True)
                                              for i in ids])

    inner = _Inner(complexes)
    ids = [4, 5, 6]
    raw = inner.batch(ids)
    masked = MaskedMetalCache(inner).batch(ids)

    is_metal = raw["node_feat"][:, 2] > 0.5
    assert int(is_metal.sum()) == len(ids), "expected one metal per complex"
    # every metal now carries the same token ...
    assert torch.equal(masked["z_idx"][is_metal],
                       torch.full((len(ids),), GENERIC_LN, dtype=torch.long))
    # ... and no non-metal atom was touched
    assert torch.equal(masked["z_idx"][~is_metal], raw["z_idx"][~is_metal])
    # geometry is untouched: distances and filtration values must survive
    assert torch.equal(masked["node_feat"], raw["node_feat"])
    assert torch.equal(masked["edge_filt"], raw["edge_filt"])
