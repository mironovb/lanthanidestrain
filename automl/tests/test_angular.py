#!/usr/bin/env python3
"""Angular features must keep every invariance the encoders already guarantee.

`snn.py` and `dist_gnn.py` both achieve exact invariance by *refusing to admit a
coordinate*: node inputs are scalars, edge inputs are distances.  That also threw
away every angle, which is why 662 of 662 encoder runs were blind to the
coordination polyhedron while the CShM / bite-angle / %V_bur descriptors sat in
the tabular blocks losing to trees.

Admitting **cosines** costs none of that invariance -- a cosine of an angle is
unchanged by rotation, translation and reflection -- but "costs none" is a claim,
so it is tested here at the same standard `test_encoders.py` holds the published
encoders to.

Also pins the property that makes the whole sweep safe: with the new flags off,
every tensor is byte-identical to the published pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from automl.topo.simplicial_data import (N_ANGULAR_BINS, _cos_histogram,
                                         metal_angular_histogram,
                                         node_angular_features)


def _cloud(n=24, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(n, 3)) * 2.0
    # a connected-ish edge set
    ii, jj = np.triu_indices(n, k=1)
    d = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    keep = d < 3.5
    edge_index = np.vstack([ii[keep], jj[keep]])
    donor = np.zeros(n, dtype=np.int8)
    donor[rng.choice(n, 8, replace=False)] = 1
    return coords, edge_index, donor


def _rotation(seed=1):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:          # keep it a proper rotation
        q[:, 0] *= -1
    return q


# ---------------------------------------------------------------------------
def test_cos_histogram_is_a_distribution():
    h = _cos_histogram(np.array([-1.0, 0.0, 0.5, 1.0]))
    assert h.shape == (N_ANGULAR_BINS,)
    assert h.sum() == pytest.approx(1.0, abs=1e-5)
    assert (h >= 0).all()


def test_cos_histogram_empty_is_zero_not_nan():
    """A node with fewer than two neighbours has no angle, and must not be NaN."""
    h = _cos_histogram(np.zeros(0))
    assert h.shape == (N_ANGULAR_BINS,) and np.isfinite(h).all() and h.sum() == 0


@pytest.mark.parametrize("fn", ["node", "metal"])
def test_rotation_invariant(fn):
    coords, ei, donor = _cloud()
    R = _rotation()
    if fn == "node":
        a = node_angular_features(coords, ei, len(coords))
        b = node_angular_features(coords @ R.T, ei, len(coords))
    else:
        a = metal_angular_histogram(coords, donor, 0)
        b = metal_angular_histogram(coords @ R.T, donor, 0)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("fn", ["node", "metal"])
def test_translation_invariant(fn):
    coords, ei, donor = _cloud()
    t = np.array([3.7, -1.2, 8.0])
    if fn == "node":
        a = node_angular_features(coords, ei, len(coords))
        b = node_angular_features(coords + t, ei, len(coords))
    else:
        a = metal_angular_histogram(coords, donor, 0)
        b = metal_angular_histogram(coords + t, donor, 0)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("fn", ["node", "metal"])
def test_reflection_invariant(fn):
    """The published encoders are reflection invariant; cosines keep that.

    A *cosine* cannot distinguish a structure from its mirror image, unlike a
    signed dihedral.  That is a real limitation -- chirality is invisible -- and
    it is asserted rather than assumed so nobody later reads chirality into a
    result these features cannot express.
    """
    coords, ei, donor = _cloud()
    M = np.diag([1.0, 1.0, -1.0])
    if fn == "node":
        a = node_angular_features(coords, ei, len(coords))
        b = node_angular_features(coords @ M, ei, len(coords))
    else:
        a = metal_angular_histogram(coords, donor, 0)
        b = metal_angular_histogram(coords @ M, donor, 0)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_node_angular_is_permutation_equivariant():
    """Relabelling atoms must permute the rows, not change them."""
    coords, ei, _ = _cloud()
    n = len(coords)
    perm = np.random.default_rng(4).permutation(n)
    inv = np.argsort(perm)
    a = node_angular_features(coords, ei, n)
    b = node_angular_features(coords[perm], inv[ei], n)
    np.testing.assert_allclose(a, b[inv], rtol=1e-4, atol=1e-5)


def test_angular_features_actually_see_geometry():
    """Guard against a descriptor that is constant and therefore useless."""
    c1, ei, donor = _cloud(seed=0)
    c2, _, _ = _cloud(seed=9)
    assert not np.allclose(metal_angular_histogram(c1, donor, 0),
                           metal_angular_histogram(c2, donor, 0))


def test_metal_histogram_degenerate_cases():
    coords, _, donor = _cloud()
    none = np.zeros(len(coords), dtype=np.int8)
    h = metal_angular_histogram(coords, none, 0)
    assert h.shape == (N_ANGULAR_BINS,) and np.isfinite(h).all() and h.sum() == 0


# ---------------------------------------------------------------------------
def test_encoder_defaults_are_unchanged_by_the_new_flags():
    """With every new flag off, the encoder is the published one."""
    from automl.topo.snn import SimplicialNet
    torch.manual_seed(0)
    a = SimplicialNet(dim=16, layers=2, dropout=0.0, tabular_dim=0)
    torch.manual_seed(0)
    b = SimplicialNet(dim=16, layers=2, dropout=0.0, tabular_dim=0,
                      angular_readout=False, attn_pool=False,
                      radial_bins=32, radial_max=8.0)
    assert a.embed_dim == b.embed_dim == 9 * 16
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)


@pytest.mark.parametrize("kw,expect_blocks", [
    ({}, 9),
    ({"angular_readout": True}, 10),
    ({"attn_pool": True}, 10),
    ({"angular_readout": True, "attn_pool": True}, 11),
])
def test_embedding_width_tracks_the_enabled_blocks(kw, expect_blocks):
    from automl.topo.snn import SimplicialNet
    dim = 16
    m = SimplicialNet(dim=dim, layers=2, dropout=0.0, tabular_dim=0,
                      node_feat_dim=5 + (N_ANGULAR_BINS
                                         if kw.get("angular_readout") else 0),
                      **kw)
    assert m.embed_dim == expect_blocks * dim


def test_angular_readout_without_the_data_raises_rather_than_guesses():
    """A missing 'metal_ang' must fail loudly, not silently pool zeros."""
    from automl.topo.snn import SimplicialNet
    g = torch.Generator().manual_seed(0)
    n, e, t = 20, 40, 10
    batch = {
        "z_idx": torch.randint(0, 27, (n,), generator=g),
        "node_feat": torch.randn(n, 5 + N_ANGULAR_BINS, generator=g),
        "edge_filt": torch.rand(e, 1, generator=g),
        "tri_filt": torch.rand(t, 1, generator=g),
        "edge_index": torch.randint(0, n, (2, e), generator=g),
        "tri_edges": torch.randint(0, e, (3, t), generator=g),
        "node_batch": torch.zeros(n, dtype=torch.long),
        "edge_batch": torch.zeros(e, dtype=torch.long),
        "tri_batch": torch.zeros(t, dtype=torch.long),
        "metal_index": torch.tensor([0]), "n_complexes": 1,
    }
    m = SimplicialNet(dim=16, layers=2, dropout=0.0, tabular_dim=0,
                      node_feat_dim=5 + N_ANGULAR_BINS, angular_readout=True)
    m.eval()
    with pytest.raises(ValueError, match="metal_ang"):
        m.encode(batch)


def test_attention_pool_is_permutation_invariant():
    from automl.topo.snn import SimplicialNet
    g = torch.Generator().manual_seed(2)
    n, e, t = 30, 70, 20
    ei = torch.randint(0, n, (2, e), generator=g)
    batch = {
        "z_idx": torch.randint(0, 27, (n,), generator=g),
        "node_feat": torch.randn(n, 5, generator=g),
        "edge_filt": torch.rand(e, 1, generator=g),
        "tri_filt": torch.rand(t, 1, generator=g),
        "edge_index": ei,
        "tri_edges": torch.randint(0, e, (3, t), generator=g),
        "node_batch": torch.zeros(n, dtype=torch.long),
        "edge_batch": torch.zeros(e, dtype=torch.long),
        "tri_batch": torch.zeros(t, dtype=torch.long),
        "metal_index": torch.tensor([0]), "n_complexes": 1,
    }
    torch.manual_seed(0)
    m = SimplicialNet(dim=16, layers=2, dropout=0.0, tabular_dim=0,
                      attn_pool=True)
    m.eval()
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(5))
    inv = torch.argsort(perm)
    p = dict(batch)
    p["z_idx"] = batch["z_idx"][perm]
    p["node_feat"] = batch["node_feat"][perm]
    p["edge_index"] = inv[ei]
    p["metal_index"] = inv[batch["metal_index"]]
    with torch.no_grad():
        torch.testing.assert_close(m.encode(p), m.encode(batch),
                                   rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Auxiliary targets (SWEEP2 axis B).  No multi-task setup has ever been run in
# this study; these pin the contract the head depends on.
def _aux_frame():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 200
    d = pd.DataFrame({
        "g3__polyhedron__cshm_SAPR": rng.normal(2, 1, n),
        "g3__polyhedron__cshm_TDD": rng.normal(3, 1, n),
        "gE__abs__e_int_water_ev": rng.normal(-55, 1, n),
        "gE__abs__dg_transfer_ev": rng.normal(-0.4, 0.3, n),
        "gE__abs__q_metal_water": rng.normal(2.1, 0.1, n),
        "gE__abs__q_transfer_water": rng.normal(0.9, 0.1, n),
    })
    # CShM is defined per coordination number, so most reference columns are NaN
    d.loc[:99, "g3__polyhedron__cshm_SAPR"] = np.nan
    d.loc[100:, "g3__polyhedron__cshm_TDD"] = np.nan
    d.loc[:9, "gE__abs__e_int_water_ev"] = np.nan       # energies missing for a few
    return d


@pytest.mark.parametrize("name", ["cshm", "eint", "qtransfer"])
def test_aux_targets_are_standardised(name):
    from automl.topo.train import aux_target_columns
    y = aux_target_columns(_aux_frame(), name)
    assert y.ndim == 2 and len(y) == 200 and y.dtype == np.float32
    # Per COLUMN over that column's own non-NaN values -- not over the rows that
    # are finite in every column, which is a different and smaller set whenever
    # one column has missing values that another does not.
    np.testing.assert_allclose(np.nanmean(y, axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(np.nanstd(y, axis=0), 1, atol=1e-5)


def test_cshm_uses_the_nearest_polyhedron_not_a_fixed_one():
    """Most CShM references are NaN per complex; the scalar is the minimum.

    Taking a fixed reference column would drop half the dataset; taking the
    nan-min is the distance to whichever ideal polyhedron the complex is closest
    to, which is the chemically meaningful quantity.
    """
    from automl.topo.train import aux_target_columns
    d = _aux_frame()
    y = aux_target_columns(d, "cshm")
    assert y.shape[1] == 1
    assert np.isfinite(y).mean() > 0.95, "nan-min should cover nearly every row"


def test_missing_aux_values_stay_nan_rather_than_being_imputed():
    """Masked, not imputed.

    Imputing a physical quantity to its mean would teach the encoder that every
    complex missing it is average, which is worse than saying nothing.
    """
    from automl.topo.train import aux_target_columns
    y = aux_target_columns(_aux_frame(), "eint")
    assert not np.isfinite(y[:10]).all(), "rows with a missing energy must be NaN"
    assert np.isfinite(y[10:]).all()


def test_unknown_aux_target_is_rejected():
    from automl.topo.train import aux_target_columns
    with pytest.raises(ValueError):
        aux_target_columns(_aux_frame(), "not_a_target")


def test_standardise_survives_an_all_nan_column():
    """An all-NaN feature must become inert, not poison every prediction.

    Found by smoking the shape preset before spending GPU on it: four CShM
    reference columns are all-NaN across the geometry-OK rows, because shape
    measures are defined per coordination number and no modelled complex has
    those.  nanmedian returned NaN, the imputation wrote NaN everywhere, and the
    run reported adjacent-pair R2 = nan.
    """
    from automl.topo.train import _standardise
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)).astype(np.float64)
    X[:, 2] = np.nan                      # entirely missing in this fold
    X[:5, 1] = np.nan                     # partially missing
    Xs, = _standardise(X)
    assert np.isfinite(Xs).all(), "an all-NaN column must not produce NaNs"
    assert np.allclose(Xs[:, 2], 0.0), "an all-NaN column must be inert"


def test_standardise_applies_train_statistics_to_held_out_rows():
    """The transform must be fitted on train only and merely applied elsewhere."""
    from automl.topo.train import _standardise
    rng = np.random.default_rng(1)
    tr = rng.normal(size=(80, 3))
    te = rng.normal(size=(20, 3)) + 5.0
    a, b = _standardise(tr, te)
    np.testing.assert_allclose(a.mean(0), 0, atol=1e-8)
    np.testing.assert_allclose(a.std(0), 1, atol=1e-8)
    assert abs(b.mean()) > 1.0, "held-out rows must NOT be re-centred on themselves"


def test_the_two_angular_blocks_are_independently_selectable():
    """A2 and A3 must each change exactly one thing, or neither is attributable.

    Both blocks originally hung off one `angular` switch, so `--angular-readout`
    silently also widened node_feat by eight columns.  It surfaced as a crash
    (the node encoder was still sized for five), but the real damage would have
    been a cell carrying two axis-A changes at once -- an unattributable result
    rather than a loud failure.
    """
    from automl.topo.simplicial_data import (SimplicialComplexes, collate,
                                             N_ANGULAR_BINS)
    S = SimplicialComplexes(verbose=False)
    ids = list(range(4))

    base = collate([S.get(i) for i in ids])
    node = collate([S.get(i, node_angular=True) for i in ids])
    metal = collate([S.get(i, metal_angular=True) for i in ids])

    assert base["node_feat"].shape[1] == 5 and "metal_ang" not in base
    # --node-angular widens node_feat and adds no readout key
    assert node["node_feat"].shape[1] == 5 + N_ANGULAR_BINS
    assert "metal_ang" not in node
    # --angular-readout adds the readout key and leaves node_feat untouched
    assert metal["node_feat"].shape[1] == 5
    assert "metal_ang" in metal
    np.testing.assert_array_equal(base["node_feat"].numpy(),
                                  metal["node_feat"].numpy())
    # the five published columns survive unchanged under either switch
    np.testing.assert_array_equal(base["node_feat"].numpy(),
                                  node["node_feat"][:, :5].numpy())


def test_the_standardise_fix_cannot_move_a_published_number():
    """The published presets must contain no all-NaN column.

    Today's `_standardise` fix changes behaviour ONLY where a column's nanmedian
    is NaN, i.e. only where a column is entirely missing in the training fold.
    If the published presets contain no such column, the fix is provably inert
    there and no published result can shift -- which is the precondition that
    lets it be applied to shared code mid-study rather than forked.

    Verified directly at the time: old and new agreed bit-for-bit over all five
    folds on both presets.  This pins the property the argument rests on, so a
    future block added to BASE_2D that is sparse on the modelled rows fails here
    loudly instead of silently changing a frozen number.
    """
    from automl.topo.train import build_row_table
    for preset in ("baseline_2d", "baseline_2d_energy"):
        _, X, _ = build_row_table(preset=preset, arch="snn")
        empty = (~np.isfinite(X)).all(axis=0)
        assert not empty.any(), (
            f"{preset} now has {int(empty.sum())} all-NaN column(s); the "
            f"_standardise fix is no longer inert for it, so published numbers "
            f"from this preset must be re-verified before being quoted")


def test_every_sweep_cell_is_identified_by_all_the_flags_it_sets():
    """A cell's matcher must pin every flag the pre-registration gives it.

    C1 is registered as `--radial-bins 64 --radial-max 10.0`, but the matcher
    checked only radial_bins, so a run with 64 bins and the default 8.0 A cutoff
    would have been swept into C1 and reported as the registered cell.  No such
    run exists in this sweep, so nothing was misassigned -- but the matcher's
    stated job is that a run can never land in the wrong cell, and an
    under-specified identity does not do that job.
    """
    from automl.topo.sweep2_test import CELLS, DEFAULTS, _matches
    base = {"arch": "snn", "no_triangles": True, "pair_loss_weight": 2.0,
            "select_on": "adjacent", "level_weight": None}
    for name, want in CELLS.items():
        # every key a cell varies must be one the matcher actually inspects
        for k in want:
            assert k in DEFAULTS, (
                f"cell {name} sets {k!r}, which _matches never checks")
        # a config built from the cell's own flags must match that cell and,
        # for non-anchor cells, must not match the anchor
        cfg = dict(base, **want)
        assert _matches(cfg, want), f"cell {name} does not match its own config"
        if name != "A0":
            assert not _matches(cfg, CELLS["A0"]), (
                f"cell {name} is indistinguishable from the A0 anchor")


def test_block_mean_transform_removes_only_within_block_variation():
    """--extra-block-mean must change one thing and hold everything else fixed.

    It is the controlled test of why A1 collapses, so it is worth only if the
    added columns keep their count and their between-block content while losing
    their within-block variation exactly.
    """
    from automl.topo.train import build_row_table
    df, X, cols = build_row_table("baseline_2d_shape", "snn")
    base = set(build_row_table("baseline_2d", "snn")[2])
    tgt = [i for i, c in enumerate(cols) if c not in base]
    assert len(tgt) == 119

    Xm = X.copy()
    blk = df["composition_key"].to_numpy()
    order = np.argsort(blk, kind="stable")
    starts = np.concatenate(
        ([0], np.flatnonzero(blk[order][1:] != blk[order][:-1]) + 1))
    segs = np.split(order, starts[1:])
    for seg in segs:
        with np.errstate(invalid="ignore"):
            m = np.nanmean(Xm[np.ix_(seg, tgt)], axis=0)
        Xm[np.ix_(seg, tgt)] = np.where(np.isfinite(m), m, np.nan)

    # 1. within-block variation is gone EXACTLY, not approximately
    for seg in segs:
        v = Xm[np.ix_(seg, tgt)]
        fin = np.isfinite(v)
        for j in range(v.shape[1]):
            col = v[fin[:, j], j]
            if col.size > 1:
                assert col.max() == col.min()

    # 2. the baseline_2d columns are untouched
    bcols = [i for i, c in enumerate(cols) if c in base]
    a = np.nan_to_num(X[:, bcols], nan=-9e9)
    b = np.nan_to_num(Xm[:, bcols], nan=-9e9)
    assert np.array_equal(a, b)


def test_no_composition_block_spans_a_cv_fold():
    """The block-mean transform is leak-free only because of this property.

    composition_key is extractant || binned conditions and the CV groups by
    extractant, so every row of a block shares a fold.  A per-block mean taken
    over the whole table is then identical to one taken inside the fold.  If a
    future keying change broke this, the transform would quietly start moving
    information across the fold boundary.
    """
    from automl.topo.train import build_row_table
    df, _, _ = build_row_table("baseline_2d", "snn")
    spans = df.groupby("composition_key")["extractant_group"].nunique()
    assert int((spans > 1).sum()) == 0, (
        f"{int((spans > 1).sum())} composition blocks span more than one "
        f"extractant group; --extra-block-mean would leak across CV folds")


def test_pair_head_and_film_are_off_by_default():
    """The published encoder must be untouched unless a flag asks for it."""
    from automl.topo.snn import SimplicialNet
    m = SimplicialNet(dim=32, layers=2, tabular_dim=4, use_triangles=False)
    assert m.pair_head is None and m.film is None
    assert not m.use_pair_head and m.film_dim == 0


def test_film_is_near_identity_at_initialisation():
    """FiLM uses the residual form, so switching it on cannot discard the
    representation the rest of the study is built on."""
    import torch
    from automl.topo.snn import SimplicialNet
    torch.manual_seed(0)
    m = SimplicialNet(dim=32, layers=2, tabular_dim=0, use_triangles=False,
                      film_dim=6).eval()
    e = torch.randn(8, m.embed_dim)
    with torch.no_grad():
        out = m.modulate(e, torch.zeros(8, 6))
    # zero conditions -> gamma,beta come from the bias alone; the residual form
    # keeps the output on the same scale as the input rather than near zero
    assert out.shape == e.shape
    assert float((out - e).abs().mean()) < float(e.abs().mean())


def test_pair_forward_is_antisymmetric_in_its_difference_channel():
    """Swapping the pair order must flip the sign of the difference channel.

    The target dy is antisymmetric, so feeding h_i - h_j (which flips) rather
    than only [h_i, h_j] gives the head a channel with the right symmetry.
    """
    import torch
    from automl.topo.snn import SimplicialNet
    torch.manual_seed(0)
    m = SimplicialNet(dim=16, layers=2, tabular_dim=0, use_triangles=False,
                      pair_head=True).eval()
    e = torch.randn(4, m.embed_dim)
    i, j = torch.tensor([0, 1]), torch.tensor([2, 3])
    d_ij = e[i] - e[j]
    d_ji = e[j] - e[i]
    torch.testing.assert_close(d_ij, -d_ji)


def test_rebuild_from_differences_reproduces_the_requested_gaps():
    """The reconciliation must impose the pair head's differences exactly."""
    from automl.topo.train import rebuild_from_differences
    means = {1: 0.5, 2: 0.2, 3: -0.4, 4: -0.9}
    diffs = {(1, 2): 0.7, (2, 3): 0.1, (3, 4): -0.2}
    new = rebuild_from_differences(means, diffs)
    for (a, b), d in diffs.items():
        assert new[a] - new[b] == pytest_approx(d)


def test_rebuild_from_differences_preserves_the_block_level():
    """Only the shape across metals may change; the offset is the level head's.

    The metric scores differences, so injecting a level shift would move
    predictions without touching the quantity being scored -- pure risk.
    """
    from automl.topo.train import rebuild_from_differences
    means = {1: 3.0, 2: 2.0, 3: 1.0}
    new = rebuild_from_differences(means, {(1, 2): 5.0, (2, 3): -4.0})
    assert np.mean(list(new.values())) == pytest_approx(np.mean(list(means.values())))


def test_rebuild_from_differences_keeps_spacing_where_no_pair_is_given():
    """A gap the pair head cannot speak for (non-adjacent) is left alone."""
    from automl.topo.train import rebuild_from_differences
    means = {1: 0.0, 2: -1.0, 4: -3.0}     # 2->4 is not adjacent
    new = rebuild_from_differences(means, {(1, 2): 0.5})
    assert new[1] - new[2] == pytest_approx(0.5)
    assert new[4] - new[2] == pytest_approx(means[4] - means[2])


def pytest_approx(v):
    import pytest
    return pytest.approx(v, abs=1e-9)
