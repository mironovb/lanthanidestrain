#!/usr/bin/env python3
"""Correctness guards for the water->octanol reorganisation block.

Each targets a way the block could carry a signal that is not the physics it
claims -- a mis-joined molecule, a sign flip, a leak, or redundancy with the
existing 3D block.  All are silent failures, so all are tested.

Run:  python3 -m pytest automl/tests/test_water_octanol.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from automl.qc.water_octanol_features import (CHARGE_DIR, OUT, _load, build,
                                              _shift_features)
from automl.topo.build_vr_conformers import _original_index

HAVE_CHARGES = ((CHARGE_DIR / "water").exists()
                and (CHARGE_DIR / "octanol").exists()
                and any((CHARGE_DIR / "water").glob("*.npz")))
needs = pytest.mark.skipif(not HAVE_CHARGES,
                           reason="conformer charges not built")


# ---------------------------------------------------------------------------
@needs
def test_same_molecule_in_both_solvents():
    """A build_id's water and octanol npz must be the same molecule.

    Same atom count, element sequence and total charge; only coordinates and
    charges differ.  A mis-joined id would pair two different molecules and the
    'reorganisation' would be nonsense that happened to correlate with size.
    """
    W = {p.stem: p for p in (CHARGE_DIR / "water").glob("*.npz")}
    O = {p.stem: p for p in (CHARGE_DIR / "octanol").glob("*.npz")}
    common = sorted(set(W) & set(O))[:40]
    assert common
    for stem in common:
        sw, xw, _ = _load(W[stem])
        so, xo, _ = _load(O[stem])
        assert sw == so, f"{stem}: element sequence differs between solvents"
        assert xw.shape == xo.shape
        # genuinely different geometries, not a duplicate file
        assert not np.allclose(xw, xo), f"{stem}: identical coords in both solvents"


@needs
def test_sign_convention_negates_signed_features_only():
    """Swapping the solvent order must negate signed features and leave
    magnitude features unchanged.

    This is the test that the octanol-water direction is applied consistently:
    a signed feature computed the wrong way round would silently flip the sign
    of the electronic-response signal for a subset of complexes.
    """
    fwd = build(limit=60).set_index("geometry_feature_build_id")
    rev = build(limit=60, swap=True).set_index("geometry_feature_build_id")
    common = fwd.index.intersection(rev.index)
    assert len(common) >= 20
    signed = ["wo_donor_dshift_signed_mean", "wo_metal_dq", "wo_donor_dq_mean",
              "wo_rg_change"]
    mag = ["wo_donor_dshift_mean", "wo_donor_dshift_max", "wo_shell_rmsd",
           "wo_angle_change_mean", "wo_heavy_rmsd", "wo_donor_dq_absmean"]
    for c in signed:
        assert np.allclose(fwd.loc[common, c], -rev.loc[common, c], atol=1e-6), c
    for c in mag:
        assert np.allclose(fwd.loc[common, c], rev.loc[common, c], atol=1e-6), c


@needs
def test_shell_rmsd_is_smaller_than_the_whole_complex():
    """The coordination shell is more rigid than the ligand periphery.

    A shell RMSD larger than the heavy-atom RMSD means the Kabsch point
    correspondence is broken (donors matched by distance rank rather than atom
    identity) -- the bug that first produced a 2.25 A median shell RMSD.
    """
    df = build(limit=120)
    ok = df["wo_heavy_rmsd"].notna()
    assert (df.loc[ok, "wo_shell_rmsd"] <= df.loc[ok, "wo_heavy_rmsd"] + 1e-6).mean() > 0.9


@needs
def test_features_are_finite_and_nondegenerate():
    df = build(limit=120).drop(columns=["geometry_feature_build_id"])
    assert np.isfinite(df.select_dtypes("number").to_numpy()).all()
    # the block must actually vary -- a constant column carries no signal and
    # would silently pad the design matrix
    varying = (df.select_dtypes("number").std() > 1e-9).sum()
    assert varying >= 15, f"only {varying} features vary"


@needs
def test_block_is_not_recoverable_from_the_single_solvent_3d_block():
    """The novelty claim (A3 - A1 in the pre-registration): the water<->octanol
    response is not what the existing single-solvent feat3d block already says.

    If the reorganisation block were linearly recoverable from feat3d, then any
    apparent gain over feat3d would be an artefact of the learner, not new
    information.  Reconstructing it from feat3d must fail (low held-out R2).
    """
    from automl.matrix_cache import load_cache
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict

    wo = build(limit=400).set_index("geometry_feature_build_id")
    df, _, _ = load_cache()
    key = df["geometry_feature_build_id"].astype(str)
    feat3d = [c for c in df.columns if c.startswith("feat3d__")]
    src = (df.assign(_k=key).drop_duplicates("_k").set_index("_k")[feat3d])
    common = wo.index.intersection(src.index)
    assert len(common) >= 100
    X = np.nan_to_num(src.loc[common].to_numpy(float))
    target_cols = ["wo_shell_rmsd", "wo_donor_dshift_mean", "wo_metal_dq"]
    for t in target_cols:
        y = wo.loc[common, t].to_numpy(float)
        if y.std() < 1e-9:
            continue
        m = make_pipeline(StandardScaler(), RidgeCV())
        p = cross_val_predict(m, X, y, cv=5)
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 < 0.6, (f"{t} is {r2:.2f}-recoverable from feat3d; the "
                          f"water<->octanol block is not new information")
