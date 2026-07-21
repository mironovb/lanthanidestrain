#!/usr/bin/env python3
"""Tests for the conformer-augmented, block-centred topological arm.

The failure modes here are all silent -- none of them raises, and each would
produce a plausible number that means something other than what it claims.  So
each is tested rather than reasoned about:

1. the conformer asset is the *same featuriser* on a different geometry;
2. nothing distinguishes an augmented structure from an original except its
   coordinates (a marker would let the model detect augmentation);
3. the row set is unchanged, so the paired bootstrap still compares like with
   like;
4. block-centring is leak-free, correctly shaped, and zero where it must be;
5. inference does not depend on how rows are grouped into batches.

Run:  python3 -m pytest automl/tests/test_conformers.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from automl.topo.simplicial_data import (CONFORMER_ROOT, ConformerComplexes,
                                         SimplicialComplexes)

HAVE_CONFORMERS = any(
    (CONFORMER_ROOT / s / "vietoris_rips_inputs.npz").exists()
    for s in ("water", "octanol"))
needs_conformers = pytest.mark.skipif(
    not HAVE_CONFORMERS,
    reason="conformer assets not built; run automl.topo.build_vr_conformers")


# ---------------------------------------------------------------------------
# 1-3. The asset itself
# ---------------------------------------------------------------------------
def test_wrapper_reports_the_same_complexes_as_the_base_asset():
    """The row set must not move.

    ``build_row_table`` keys rows off ``index_of``.  If the wrapper reported a
    different set, the conformer arm would be scored on different rows from the
    arm it is compared against and the paired bootstrap would silently be
    comparing two datasets -- the failure this study already caught once
    between the 4,746-row and 4,742-row arms.
    """
    C = ConformerComplexes(verbose=False)
    S = SimplicialComplexes(verbose=False)
    assert len(C) == len(S)
    assert C.build_ids == S.build_ids
    assert all(C.index_of(b) == S.index_of(b) for b in S.build_ids)


def test_conformer_zero_is_exactly_the_shipped_complex():
    """Conformer 0 must be bit-identical to what every published run used."""
    C = ConformerComplexes(verbose=False)
    S = SimplicialComplexes(verbose=False)
    for k in (0, 5, 100):
        a = C.get(k, conformer=0, filtration_max=3.5, heavy_only=True)
        b = S.get(k, filtration_max=3.5, heavy_only=True)
        for f in ("z_idx", "charge", "is_metal", "is_donor", "dist_to_metal",
                  "edge_index", "edge_filt"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f


@needs_conformers
def test_conformers_are_the_same_molecule_in_a_different_shape():
    """Same atoms, same elements, same coordination number -- different shape.

    A mis-joined build id would pair a complex with another molecule's geometry,
    which would look like a richly informative conformer and be entirely
    spurious.  Atom identity and coordination number must carry over.

    The donor *mask* deliberately is **not** asserted equal.  I first wrote it
    that way and it failed: ``_coordination_shell`` selects the nearest
    ``core_cn`` donor atoms by distance, so a conformer can coordinate a
    different atom.  Measured over 304 multi-conformer complexes, 94.7 % keep an
    identical mask and 5.3 % swap -- 13 of those 16 preserving the donor element
    multiset, i.e. which oxygen of a nitrate happens to be closest.  That is the
    coordination sphere genuinely moving, which is the physics the conformers
    exist to sample, not a defect.  What must not change is the donor *count*,
    because a complex whose coordination number changed is a different species
    and is dropped at build time.
    """
    C = ConformerComplexes(verbose=False)
    checked = 0
    for k in range(len(C)):
        if C.n_conformers(k) < 2:
            continue
        a = C.get(k, conformer=0, filtration_max=3.5, heavy_only=True)
        b = C.get(k, conformer=1, filtration_max=3.5, heavy_only=True)
        assert np.array_equal(a.z_idx, b.z_idx), "elements changed"
        assert np.array_equal(a.is_metal, b.is_metal), "metal moved"
        assert int(a.is_donor.sum()) == int(b.is_donor.sum()), \
            "coordination number changed -- should have been dropped at build"
        assert not np.allclose(a.dist_to_metal, b.dist_to_metal), \
            "conformer is geometrically identical to the original"
        checked += 1
        if checked >= 25:
            break
    assert checked >= 10, "too few multi-conformer complexes to test"


@needs_conformers
def test_donor_identity_rarely_moves_and_never_changes_count():
    """Quantifies the swap rate rather than leaving it as a footnote.

    If most conformers reassigned their donors, the "same complex, different
    shape" claim would be too weak to build an augmentation on. Measured, it is
    a small minority; this pins that so a future asset rebuild cannot silently
    make it common.
    """
    C = ConformerComplexes(verbose=False)
    same = diff = 0
    for k in range(0, len(C), 3):
        if C.n_conformers(k) < 2:
            continue
        a = C.get(k, 0, 3.5, True)
        b = C.get(k, 1, 3.5, True)
        assert int(a.is_donor.sum()) == int(b.is_donor.sum())
        if np.array_equal(a.is_donor, b.is_donor):
            same += 1
        else:
            diff += 1
        if same + diff >= 60:
            break
    total = same + diff
    assert total >= 30
    assert diff / total < 0.25, (
        f"{diff}/{total} conformers reassigned their donor set; the conformer "
        f"is supposed to be the same complex in a different shape")


@needs_conformers
def test_no_marker_separates_augmented_from_original():
    """``charge_missing`` must not identify the augmented structures.

    If conformers had no Mulliken charges, every one of them would carry
    ``charge_missing = 1`` and every original would carry 0 -- a free label for
    "this is an augmented sample".  The model would learn the label, and the
    gain would be an artefact.  This is why the charges were recomputed with
    single points rather than imputed.
    """
    C = ConformerComplexes(verbose=False)
    orig, aug = [], []
    for k in range(0, len(C), 7):
        if C.n_conformers(k) < 2:
            continue
        orig.append(float(C.get(k, 0, 3.5, True).charge_missing.mean()))
        aug.append(float(C.get(k, 1, 3.5, True).charge_missing.mean()))
    assert orig and aug
    assert max(aug) < 0.01, f"augmented structures carry missing charges: {max(aug)}"
    assert abs(np.mean(orig) - np.mean(aug)) < 0.01


@needs_conformers
def test_conformer_index_wraps_instead_of_raising():
    """Complexes have unequal conformer counts; sampling must not crash a fold."""
    C = ConformerComplexes(verbose=False)
    k = next(i for i in range(len(C)) if C.n_conformers(i) >= 2)
    a = C.get(k, conformer=1, filtration_max=3.5, heavy_only=True)
    b = C.get(k, conformer=1 + C.n_conformers(k) - 1, filtration_max=3.5,
              heavy_only=True)
    assert a.z_idx.shape == b.z_idx.shape
    single = next((i for i in range(len(C)) if C.n_conformers(i) == 1), None)
    if single is not None:
        # falls back to the shipped geometry rather than raising
        C.get(single, conformer=3, filtration_max=3.5, heavy_only=True)


# ---------------------------------------------------------------------------
# 4-5. Block-centring
# ---------------------------------------------------------------------------
def _centre(emb: torch.Tensor, blocks: np.ndarray) -> torch.Tensor:
    """Mirror of the implementation in train.run_fold._centre."""
    import pandas as pd
    codes, _ = pd.factorize(blocks)
    idx = torch.as_tensor(codes, dtype=torch.long)
    n = int(codes.max()) + 1
    sums = torch.zeros((n, emb.shape[1]), dtype=emb.dtype).index_add_(0, idx, emb)
    counts = torch.zeros(n, dtype=emb.dtype).index_add_(
        0, idx, torch.ones_like(idx, dtype=emb.dtype))
    means = sums / counts.clamp(min=1.0).unsqueeze(-1)
    return torch.cat([emb, emb - means[idx]], dim=-1)


def test_block_centring_doubles_width_and_preserves_the_absolute_half():
    emb = torch.randn(6, 4)
    blocks = np.array(["a", "a", "a", "b", "b", "c"])
    out = _centre(emb, blocks)
    assert out.shape == (6, 8)
    assert torch.equal(out[:, :4], emb), "the absolute embedding must survive"


def test_relative_half_is_zero_for_single_metal_blocks():
    """347 of 552 composition blocks hold one metal.

    Concatenating rather than replacing is what keeps those rows' absolute
    embedding; the relative half being exactly zero there is the property that
    makes the choice safe.
    """
    emb = torch.randn(4, 3)
    out = _centre(emb, np.array(["a", "a", "b", "c"]))
    assert torch.allclose(out[2, 3:], torch.zeros(3), atol=1e-6)
    assert torch.allclose(out[3, 3:], torch.zeros(3), atol=1e-6)
    assert not torch.allclose(out[0, 3:], torch.zeros(3), atol=1e-6)


def test_relative_half_sums_to_zero_within_every_block():
    emb = torch.randn(9, 5)
    blocks = np.array(list("aaabbbccc"))
    rel = _centre(emb, blocks)[:, 5:]
    for b in ("a", "b", "c"):
        assert torch.allclose(rel[blocks == b].sum(0), torch.zeros(5), atol=1e-5)


def test_block_centring_is_invariant_to_row_order():
    """Predictions must not depend on how rows are grouped into batches.

    ``_predict`` batches by composition block precisely so this holds; if it
    sliced blocks across batch boundaries the feature would change with the
    batch size, which is not a property of the data.
    """
    emb = torch.randn(8, 3)
    blocks = np.array(list("abababab"))
    perm = torch.as_tensor([3, 0, 7, 1, 5, 2, 6, 4])
    a = _centre(emb, blocks)[perm]
    b = _centre(emb[perm], blocks[perm.numpy()])
    assert torch.allclose(a, b, atol=1e-6)


def test_head_width_matches_the_centred_embedding():
    """A mismatch here is a shape error at fold 0; a *silent* mismatch would be
    a head sized for the wrong stream, so it is asserted rather than trusted."""
    from automl.topo.snn import SimplicialNet
    plain = SimplicialNet(dim=96, layers=1, tabular_dim=746, head_embed_mult=1)
    wide = SimplicialNet(dim=96, layers=1, tabular_dim=746, head_embed_mult=2)
    assert plain.head[1].in_features == plain.embed_dim + 746
    assert wide.head[1].in_features == 2 * wide.embed_dim + 746
    # encode() is unchanged, so a mult=1 model stays compatible with prior runs
    assert plain.embed_dim == wide.embed_dim == 9 * 96
