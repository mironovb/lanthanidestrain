"""Focused tests for feature extraction from existing extended XYZ files."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.geometry_features import (
    UNCOMPUTED_XTB_FEATURES,
    build_geometry_feature_artifacts,
    geometry_scalar_rows,
    persistence_diagram,
    persistence_image,
    read_extxyz,
)


_TMP = _REPO_ROOT / "tests" / "_tmp_geometry_features"


def _write_xyz() -> Path:
    _TMP.mkdir(parents=True, exist_ok=True)
    path = _TMP / "Eu_test_b1.xyz"
    path.write_text(
        "\n".join([
            "4",
            'Properties=species:S:1:pos:R:3:charge:R:1 energy=-10.5 '
            'free_energy=-10.4 dipole="1.0 2.0 2.0"',
            "Eu 0.0 0.0 0.0 0.80",
            "O  1.0 0.0 0.0 -0.30",
            "N  0.0 2.0 0.0 -0.20",
            "S  0.0 0.0 3.0 -0.10",
        ]) + "\n",
        encoding="utf-8",
    )
    return path


def test_extxyz_physical_and_polyhedron_values():
    geometry = read_extxyz(_write_xyz())
    physical, polyhedron, donor_indices = geometry_scalar_rows(
        "b1", geometry, "Eu", 3
    )
    assert geometry.has_xtb_properties
    assert physical["complex_total_energy_eV"] == -10.5
    assert physical["dipole_magnitude"] == 3.0
    assert physical["metal_partial_charge"] == 0.8
    assert polyhedron["ln_donor_distance_01"] == 1.0
    assert polyhedron["ln_donor_distance_02"] == 2.0
    assert polyhedron["ln_donor_distance_03"] == 3.0
    assert polyhedron["donor_angle_01_02_deg"] == 90.0
    assert donor_indices.tolist() == [1, 2, 3]
    assert all(np.isnan(physical[name]) for name in UNCOMPUTED_XTB_FEATURES)


def test_persistence_image_stays_image_native():
    geometry = read_extxyz(_write_xyz())
    image = persistence_image(persistence_diagram(geometry.coordinates))
    assert image.shape == (20, 20)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()


def test_build_artifacts_are_keyed_by_safe_exp_id():
    xyz = _write_xyz()
    out = _TMP / "blocks"
    qc = pd.DataFrame({
        "build_id": ["b1"],
        "geometry_key": ["63|CCO|water"],
        "qc_class": ["OK"],
        "xyz_path": [str(xyz)],
        "metal_symbol": ["Eu"],
        "metal_ox": [3],
        "coreCN": [3],
        "n_ligs": [1],
        "n_fill": [0],
        "fill_ligand": ["water"],
        "inner_sphere_anion": ["water"],
        "SMILES_FOR_ARCHITECTOR": ["CCO"],
        "full_method": ["GFN2-xTB"],
        "relax": [True],
        "geometry_source": ["synthetic"],
    })
    rows = pd.DataFrame({
        "safe_exp_id": ["Eu_SAFE:1", "Eu_SAFE:2"],
        "build_id": ["b1", "b1"],
        "geometry_key": ["63|CCO|water", "63|CCO|water"],
        "canonical_smiles": ["CCO", "CCO"],
    })
    result = build_geometry_feature_artifacts(
        qc, rows, repo_root=_REPO_ROOT, output_dir=out,
        pi_resolution=8, vr_max_edge=4.0,
    )
    keyed = pd.read_parquet(out / "features_by_safe_exp_id.parquet")
    assert keyed["safe_exp_id"].tolist() == ["Eu_SAFE:1", "Eu_SAFE:2"]
    assert keyed["safe_exp_id"].is_unique
    row_assets = result["row_asset_columns"]
    assert row_assets["complex_pi_image_index"].tolist() == [0, 0]
    assert row_assets["vr_graph_index"].tolist() == [0, 0]
    with np.load(out / "complex_gfn2xtb_pi_images.npz") as data:
        assert data["images"].shape == (1, 1, 8, 8)
    with np.load(out / "vietoris_rips_inputs.npz") as data:
        assert data["build_ids"].tolist() == ["b1"]
        assert data["triangle_index"].shape[0] == 3


def teardown_module():
    shutil.rmtree(_TMP, ignore_errors=True)


if __name__ == "__main__":
    tests = [
        test_extxyz_physical_and_polyhedron_values,
        test_persistence_image_stays_image_native,
        test_build_artifacts_are_keyed_by_safe_exp_id,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    teardown_module()
