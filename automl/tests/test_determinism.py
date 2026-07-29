#!/usr/bin/env python3
"""The deterministic scatter must agree with the published one, and be exact.

Two properties, and they are different claims:

1. **Agreement.** Turning determinism on must not change what the model
   computes.  If the sorted reduction disagreed with the atomic one by more than
   float32 rounding, every deterministic run would be measuring a different
   model and could not be compared with a published one.
2. **Invariance.** The deterministic path's answer must not depend on the order
   the inputs arrive in.  This is the property the atomic path lacks, and it is
   what makes a re-run reproducible.

Both are checked on CPU with synthetic tensors, so they run everywhere.  The
end-to-end GPU claim -- that two full re-runs produce bit-identical out-of-fold
vectors -- is a cluster measurement, not a unit test; it lives in
``automl/slurm/determinism.sh`` and ``DETERMINISM_RESULTS.md``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from automl.topo import snn


@pytest.fixture(autouse=True)
def _restore_flag():
    """Never leak the global into another test module."""
    before = snn._DETERMINISTIC
    yield
    snn.set_deterministic(before)


def _case(n: int = 4000, dim: int = 17, segs: int = 61, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randn(n, dim, generator=g)
    index = torch.randint(0, segs, (n,), generator=g)
    return src, index, segs


def test_deterministic_sum_agrees_with_published_path():
    src, index, segs = _case()
    snn.set_deterministic(False)
    ref = snn.scatter_sum(src, index, segs)
    snn.set_deterministic(True)
    got = snn.scatter_sum(src, index, segs)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_deterministic_mean_agrees_with_published_path():
    src, index, segs = _case(seed=3)
    snn.set_deterministic(False)
    ref = snn.scatter_mean(src, index, segs)
    snn.set_deterministic(True)
    got = snn.scatter_mean(src, index, segs)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_deterministic_sum_is_permutation_invariant_bit_for_bit():
    """The property the atomic path does not have.

    Shuffling the rows is exactly what a different thread schedule does to the
    accumulation order, so an answer that survives a shuffle bit-for-bit is one
    that cannot drift between runs.
    """
    src, index, segs = _case(seed=5)
    snn.set_deterministic(True)
    a = snn.scatter_sum(src, index, segs)
    perm = torch.randperm(src.shape[0], generator=torch.Generator().manual_seed(9))
    b = snn.scatter_sum(src[perm], index[perm], segs)
    assert torch.equal(a, b), "sorted scatter still depends on input order"


def test_published_sum_is_not_bitwise_permutation_invariant_on_cpu_float32():
    """Sanity check on the premise, so the test above is not vacuous.

    On CPU ``index_add_`` is sequential, so it *is* order-dependent in the last
    bits under a shuffle -- the same sensitivity that becomes nondeterminism on
    a GPU.  If this ever starts passing exactly, the deterministic path is
    solving a problem that no longer exists and should be re-justified.
    """
    src, index, segs = _case(seed=11, n=20000, dim=8, segs=13)
    snn.set_deterministic(False)
    a = snn.scatter_sum(src, index, segs)
    perm = torch.randperm(src.shape[0], generator=torch.Generator().manual_seed(2))
    b = snn.scatter_sum(src[perm], index[perm], segs)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)   # same number...
    assert not torch.equal(a, b)                             # ...different bits


def test_empty_and_single_segment_edge_cases():
    snn.set_deterministic(True)
    empty = snn.scatter_sum(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), 3)
    assert empty.shape == (3, 4) and torch.equal(empty, torch.zeros(3, 4))
    one = snn.scatter_sum(torch.ones(5, 2), torch.zeros(5, dtype=torch.long), 2)
    torch.testing.assert_close(one, torch.tensor([[5.0, 5.0], [0.0, 0.0]]))


def test_missing_segments_stay_zero():
    """A segment with no members must be 0, not left at an accumulator value."""
    snn.set_deterministic(True)
    src = torch.ones(6, 3)
    index = torch.tensor([0, 0, 2, 2, 4, 4])
    out = snn.scatter_sum(src, index, 5)
    assert torch.equal(out[1], torch.zeros(3))
    assert torch.equal(out[3], torch.zeros(3))
    torch.testing.assert_close(out[0], torch.full((3,), 2.0))


def test_scatter_max_unchanged_by_the_flag():
    """Max needs no deterministic variant; assert the flag does not touch it."""
    src, index, segs = _case(seed=7)
    snn.set_deterministic(False)
    a = snn.scatter_max(src, index, segs)
    snn.set_deterministic(True)
    b = snn.scatter_max(src, index, segs)
    assert torch.equal(a, b)


def test_encoder_output_matches_between_modes():
    """The property that matters: same model, same answer, either mode."""
    torch.manual_seed(0)
    model = snn.SimplicialNet(dim=16, layers=2, dropout=0.0, tabular_dim=0)
    model.eval()
    n_nodes, n_edges, n_tris = 40, 90, 55
    g = torch.Generator().manual_seed(1)
    batch = {
        "z_idx": torch.randint(0, 27, (n_nodes,), generator=g),
        "node_feat": torch.randn(n_nodes, 5, generator=g),
        "edge_filt": torch.rand(n_edges, 1, generator=g),
        "tri_filt": torch.rand(n_tris, 1, generator=g),
        "edge_index": torch.randint(0, n_nodes, (2, n_edges), generator=g),
        "tri_edges": torch.randint(0, n_edges, (3, n_tris), generator=g),
        "node_batch": torch.randint(0, 2, (n_nodes,), generator=g).sort().values,
        "edge_batch": torch.randint(0, 2, (n_edges,), generator=g).sort().values,
        "tri_batch": torch.randint(0, 2, (n_tris,), generator=g).sort().values,
        "metal_index": torch.tensor([0, 20]),
        "n_complexes": 2,
    }
    with torch.no_grad():
        snn.set_deterministic(False)
        ref = model.encode(batch)
        snn.set_deterministic(True)
        got = model.encode(batch)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    assert np.isfinite(got.numpy()).all()
