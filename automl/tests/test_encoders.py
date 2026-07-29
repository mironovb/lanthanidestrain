#!/usr/bin/env python3
"""The two new encoders must satisfy the same invariances the SNN does.

``snn.py`` claims exact invariance to rotation, translation, reflection and atom
permutation -- by construction, because no raw coordinate enters and every
aggregation is scatter-based -- and asserts it in the tests.  A control encoder
that quietly lacked one of those would lose the comparison for a reason having
nothing to do with simplicial structure, which is precisely the confound the
whole third-encoder experiment exists to avoid.

So the same properties are asserted here for:

* ``DistanceNet``          -- the no-simplex control (continuous-filter messages)
* ``SimplicialNet(use_triangles=False)`` -- the no-2-simplex ablation

Plus the two structural claims the experiment rests on: that the ablation really
does remove the triangle pathway, and that both encoders emit the same embedding
width as the full SNN so the head cannot differ in capacity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from automl.topo.dist_gnn import DistanceNet
from automl.topo.snn import SimplicialNet


def _batch(n_nodes=48, n_edges=120, n_tris=70, seed=0):
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, n_nodes, (2, n_edges), generator=g)
    return {
        "z_idx": torch.randint(0, 27, (n_nodes,), generator=g),
        "node_feat": torch.randn(n_nodes, 5, generator=g),
        "edge_filt": torch.rand(n_edges, 1, generator=g) * 3.5,
        "tri_filt": torch.rand(n_tris, 1, generator=g) * 3.5,
        "edge_index": ei,
        "tri_edges": torch.randint(0, n_edges, (3, n_tris), generator=g),
        "node_batch": torch.zeros(n_nodes, dtype=torch.long),
        "edge_batch": torch.zeros(n_edges, dtype=torch.long),
        "tri_batch": torch.zeros(n_tris, dtype=torch.long),
        "metal_index": torch.tensor([0]),
        "n_complexes": 1,
    }


def _models(dim=24, layers=2):
    torch.manual_seed(0)
    full = SimplicialNet(dim=dim, layers=layers, dropout=0.0, tabular_dim=0)
    torch.manual_seed(0)
    notri = SimplicialNet(dim=dim, layers=layers, dropout=0.0, tabular_dim=0,
                          use_triangles=False)
    torch.manual_seed(0)
    dist = DistanceNet(dim=dim, layers=layers, dropout=0.0, tabular_dim=0)
    for m in (full, notri, dist):
        m.eval()
    return full, notri, dist


@pytest.mark.parametrize("which", ["notri", "dist"])
def test_embedding_width_matches_the_full_snn(which):
    """Equal width, so no capacity difference is smuggled into the comparison."""
    full, notri, dist = _models()
    m = {"notri": notri, "dist": dist}[which]
    assert m.embed_dim == full.embed_dim
    with torch.no_grad():
        assert m.encode(_batch()).shape == full.encode(_batch()).shape


@pytest.mark.parametrize("which", ["notri", "dist"])
def test_permutation_invariant(which):
    """Relabelling the atoms must not change the complex-level embedding."""
    full, notri, dist = _models()
    m = {"notri": notri, "dist": dist}[which]
    b = _batch()
    n = b["node_feat"].shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(4))
    inv = torch.argsort(perm)
    p = dict(b)
    p["z_idx"] = b["z_idx"][perm]
    p["node_feat"] = b["node_feat"][perm]
    p["edge_index"] = inv[b["edge_index"]]
    p["metal_index"] = inv[b["metal_index"]]
    with torch.no_grad():
        torch.testing.assert_close(m.encode(p), m.encode(b),
                                   rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("which", ["notri", "dist"])
def test_edge_direction_invariant(which):
    """An edge is unordered; swapping i and j must change nothing.

    ``MPSNLayer`` documents a real bug of exactly this kind -- two separately
    normalised per-endpoint means made the answer depend on an arbitrary row
    assignment, and differed by 2.6e-4 in both float32 and float64, so it was not
    rounding.  ``CFConvLayer`` is written to avoid it; this is the check.
    """
    full, notri, dist = _models()
    m = {"notri": notri, "dist": dist}[which]
    b = _batch()
    flip = dict(b)
    flip["edge_index"] = b["edge_index"].flip(0)
    with torch.no_grad():
        torch.testing.assert_close(m.encode(flip), m.encode(b),
                                   rtol=1e-4, atol=1e-4)


def test_no_triangle_ablation_ignores_the_triangle_inputs():
    """The ablation must be a real removal, not a reweighting.

    Feeding completely different triangles must leave the answer untouched; if
    it did not, the arm would still be reading 2-simplex information and the
    'is it simplicial?' comparison would be answering nothing.
    """
    _, notri, _ = _models()
    b = _batch()
    other = _batch(seed=99)
    b2 = dict(b)
    b2["tri_filt"] = other["tri_filt"]
    b2["tri_edges"] = other["tri_edges"]
    b2["tri_batch"] = other["tri_batch"]
    with torch.no_grad():
        assert torch.equal(notri.encode(b2), notri.encode(b))


def test_full_snn_does_read_the_triangles():
    """Guard the test above: if the full model ignored them too, it is vacuous."""
    full, _, _ = _models()
    b = _batch()
    other = _batch(seed=99)
    b2 = dict(b)
    b2["tri_filt"] = other["tri_filt"]
    b2["tri_edges"] = other["tri_edges"]
    with torch.no_grad():
        assert not torch.allclose(full.encode(b2), full.encode(b),
                                  rtol=1e-6, atol=1e-6)


def test_dist_net_reads_distance_not_identity():
    """Perturbing the interatomic distances must move the embedding."""
    _, _, dist = _models()
    b = _batch()
    b2 = dict(b)
    b2["edge_filt"] = b["edge_filt"] * 0.5
    with torch.no_grad():
        assert not torch.allclose(dist.encode(b2), dist.encode(b),
                                  rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("which", ["notri", "dist"])
def test_outputs_are_finite_with_isolated_nodes(which):
    """A node with no incident edge must not produce NaN through a mean."""
    full, notri, dist = _models()
    m = {"notri": notri, "dist": dist}[which]
    b = _batch(n_nodes=30, n_edges=10, n_tris=4)
    # force the last 15 nodes to be isolated
    b["edge_index"] = torch.randint(0, 15, (2, 10))
    with torch.no_grad():
        out = m.encode(b)
    assert np.isfinite(out.numpy()).all()


def test_dist_net_forward_with_tabular_block():
    torch.manual_seed(0)
    m = DistanceNet(dim=16, layers=2, dropout=0.0, tabular_dim=7)
    m.eval()
    with torch.no_grad():
        y = m(_batch(), torch.randn(1, 7))
    assert y.shape == (1,) and np.isfinite(y.numpy()).all()
