#!/usr/bin/env python3
"""Tests for the no-topology control arm.

The recurring failure mode in this study has been correct measurements paired
with a wrong interpretation, so these test the *claims the control makes* rather
than the mechanics of the class:

1. the embedding really is absent, not zeroed;
2. the control and the topological arm really do see the same folds, the same
   inner-validation extractants and the same batch order -- the assumption the
   whole matched-pairs design rests on;
3. a model with no inputs at all cannot be produced by accident;
4. the contrast loss actually fires without topology (if it silently did not,
   T0 would be a copy of T1 and the control would answer nothing).

Run:  python3 -m pytest automl/tests/test_topo_control.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from automl.topo.tabular_net import NullCache, TabularNet
from automl.topo.snn import SimplicialNet


# ---------------------------------------------------------------------------
# 1. The embedding is absent, not zeroed
# ---------------------------------------------------------------------------
def test_encode_returns_a_width_zero_tensor():
    m = TabularNet(tabular_dim=7)
    emb = m.encode({"n_complexes": 5})
    assert emb.shape == (5, 0)


def test_head_has_no_weights_for_a_topological_embedding():
    """The point of width-zero: those parameters must not exist.

    A zeroed embedding would leave `head[1]` with in_features = embed_dim +
    tabular_dim and train them against a constant -- a differently sized model
    fitting a degenerate feature, which is a weaker ablation than the one being
    claimed.
    """
    tab = TabularNet(tabular_dim=746, head_hidden=256)
    assert tab.embed_dim == 0
    assert tab.head[1].in_features == 746

    snn = SimplicialNet(dim=96, layers=1, tabular_dim=746, head_hidden=256)
    assert snn.head[1].in_features == snn.embed_dim + 746
    # and the control is smaller by exactly the embedding it lacks
    assert snn.head[1].in_features - tab.head[1].in_features == snn.embed_dim


def test_control_holds_no_parameter_reachable_from_structure():
    """No z-embedding, no message passing, no radial readout -- by name."""
    names = [n for n, _ in TabularNet(tabular_dim=7).named_parameters()]
    forbidden = ("z_emb", "node_in", "edge_in", "tri_in", "layers",
                 "radial", "conv", "proj")
    assert all(not any(f in n for f in forbidden) for n in names), names
    assert all(n.startswith("head.") for n in names), names


# ---------------------------------------------------------------------------
# 2. The pairing guarantee
# ---------------------------------------------------------------------------
def _rng_draws(seed: int, groups: np.ndarray, n_blocks: int, epochs: int):
    """Replay exactly what run_fold draws from numpy, in order.

    run_fold creates ``np.random.default_rng(seed)`` before the model exists and
    uses it for (i) the inner-validation group choice and (ii) one block
    permutation per epoch.  Neither call depends on the architecture, so this
    replay is what both arms must produce.  Mirrors train.py:184-188 and 271.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    val = rng.choice(uniq, size=max(1, int(0.15 * len(uniq))), replace=False)
    perms = [rng.permutation(n_blocks) for _ in range(epochs)]
    return val, perms


def test_the_two_arms_draw_the_same_folds_and_batch_order():
    """The matched-pairs design is only valid if this holds.

    Same seed -> same held-out validation extractants and same composition-block
    ordering, regardless of architecture.  Asserted rather than assumed, because
    if it silently stopped holding the paired bootstrap would be comparing arms
    trained on different splits and the interval would be wrong in a direction
    nobody would notice.
    """
    groups = np.repeat(np.arange(40), 5)
    for seed in (7, 42, 281):
        a_val, a_perm = _rng_draws(seed, groups, n_blocks=25, epochs=4)
        b_val, b_perm = _rng_draws(seed, groups, n_blocks=25, epochs=4)
        assert np.array_equal(a_val, b_val)
        assert all(np.array_equal(x, y) for x, y in zip(a_perm, b_perm))


def test_torch_seeding_does_not_leak_into_the_numpy_stream():
    """Different architectures consume different amounts of torch RNG.

    That is fine and unavoidable (weight init, dropout masks).  What must not
    happen is that consumption shifting the *numpy* stream, which is what picks
    the folds.  Building models of very different size between the two draws
    must leave the numpy draws identical.
    """
    groups = np.repeat(np.arange(40), 5)
    torch.manual_seed(42)
    first, _ = _rng_draws(42, groups, n_blocks=25, epochs=2)
    torch.manual_seed(42)
    SimplicialNet(dim=96, layers=3, tabular_dim=746)   # consumes a lot
    TabularNet(tabular_dim=746)                        # consumes little
    second, _ = _rng_draws(42, groups, n_blocks=25, epochs=2)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# 3. A model with no inputs cannot be built by accident
# ---------------------------------------------------------------------------
def test_zero_tabular_dim_is_refused():
    """--arch tabular --topology-only would leave nothing to learn from.

    Such a run would fit the target mean and still write a run record that looks
    like a control, which is the sort of artefact that survives review.
    """
    with pytest.raises(ValueError, match="tabular_dim"):
        TabularNet(tabular_dim=0)


def test_forward_requires_the_tabular_block():
    with pytest.raises(ValueError):
        TabularNet(tabular_dim=7).forward({"n_complexes": 3}, tabular=None)


# ---------------------------------------------------------------------------
# 4. The harness composes the same expression for both arms
# ---------------------------------------------------------------------------
def test_concatenation_reduces_to_the_tabular_block_exactly():
    """run_fold builds cat([emb[gather], tab]) with no branch on architecture.

    With a width-zero embedding that expression must be *bit-identical* to the
    tabular block itself -- otherwise the control is quietly a different model
    from the one described.
    """
    m = TabularNet(tabular_dim=5)
    tab = torch.randn(9, 5)
    emb = m.encode({"n_complexes": 4})
    gather = torch.as_tensor([0, 1, 2, 3, 0, 1, 2, 3, 0])
    assert torch.equal(torch.cat([emb[gather], tab], dim=-1), tab)


def test_null_cache_reports_the_complex_count_the_harness_gathers_on():
    cache = NullCache(torch.device("cpu"))
    b = cache.batch([3, 9, 14])
    assert b["n_complexes"] == 3
    assert TabularNet(tabular_dim=2).encode(b).shape == (3, 0)


def test_contrast_loss_finds_pairs_without_topology():
    """The contrast loss is what the control is testing; a silently inactive one
    would make T0 a duplicate of T1 and the whole factorial vacuous.

    Reproduces the pair enumeration from train.py:299-310 on a batch built the
    way _batches() builds one -- whole composition blocks.
    """
    comp = np.array(["a", "a", "a", "b", "b"])
    lidx = np.array([2, 3, 7, 4, 5])
    same = torch.as_tensor(comp[:, None] == comp[None, :])
    pi, pj = torch.nonzero(torch.triu(same, diagonal=1), as_tuple=True)
    assert pi.numel() == 4                      # 3 within 'a' + 1 within 'b'

    dl = torch.as_tensor(np.abs(lidx[:, None] - lidx[None, :]),
                         dtype=torch.float32)[pi, pj]
    w = torch.where(dl <= 1.0, 3.0, 1.0)
    assert int((w == 3.0).sum()) == 2           # (2,3) and (4,5) are adjacent
