"""Regression + behaviour tests for stage E (the optional 3D-feature join).

The headline test (``test_rescued_geometry_is_clean_when_schemas_mix``) pins the
exact bug that shipped: the stage-2 accepted table (accepted_* paths, no
xyz_exists) mixed with a reports table (canonical schema) used to mark every
rescued geometry geometry_ok=False. The rest cover gating, weights, and the
final column contract.

Runnable with pytest, or standalone: ``.venv/bin/python tests/test_stage_e.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_builder():
    path = _REPO_ROOT / "scripts" / "build_dataset_no3d.py"
    spec = importlib.util.spec_from_file_location("build_dataset_no3d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b = _load_builder()


# --- fixtures: tables shaped exactly like the real producer/report outputs -----
def _reports_table() -> pd.DataFrame:
    """Canonical reports-style QC table (has xyz_exists, xyz_path, energy_eV)."""
    return pd.DataFrame({
        "build_id": ["b_rescued", "b_plain", "b_fail", "b_noxyz"],
        "qc_class": ["FAIL_LONG_BOND", "OK", "FAIL_LONG_BOND", "OK"],
        "xyz_exists": [True, True, True, False],
        "xyz_path": ["p/b_rescued.xyz", "p/b_plain.xyz", "p/b_fail.xyz", ""],
        "mol2_path": ["p/b_rescued.mol2", "p/b_plain.mol2", "p/b_fail.mol2", ""],
        "energy_eV": [-9.0, -8.0, -7.0, -6.0],
        "coreCN_max_dist": [3.4, 2.4, 3.5, 2.3],
        "next_donor_dist": [4.4, 3.4, 4.5, 3.3],
        "gap_after_coreCN": [1.0, 1.0, 1.0, 1.0],
    })


def _accepted_table() -> pd.DataFrame:
    """Stage-2 accepted table: accepted_* paths, NO xyz_exists / energy_eV."""
    return pd.DataFrame({
        "build_id": ["b_rescued"],
        "qc_class": ["OK"],
        "accepted_xyz_path": ["regen/b_rescued.xyz"],
        "accepted_mol2_path": ["regen/b_rescued.mol2"],
        "coreCN_max_dist": [2.9],
        "gap_after_coreCN": [0.9],
        "nearest_coreCN_sig": ["sig"],
        "accepted_for_clean_3d_features": [True],
    })


def _write(df: pd.DataFrame, name: str) -> Path:
    out = _REPO_ROOT / "tests" / "_tmp"
    out.mkdir(exist_ok=True)
    path = out / name
    df.to_csv(path, index=False)
    return path


def _dataset(build_ids: list[str], metals: list[str] | None = None) -> pd.DataFrame:
    metals = metals or ["La"] * len(build_ids)
    return pd.DataFrame({
        b.METAL_COL: metals,
        "safe_exp_id": [f"e{i}" for i in range(len(build_ids))],
        "build_id": build_ids,
        "geometry_key": [f"g{i}" for i in range(len(build_ids))],
        b.CANONICAL_SMILES_COL: ["CCO"] * len(build_ids),
    })


# --- the regression test ------------------------------------------------------
def test_rescued_geometry_is_clean_when_schemas_mix():
    pa = _write(_accepted_table(), "accepted.csv")
    pr = _write(_reports_table(), "reports.csv")
    qc_index, used, fallbacks = b.load_geometry_qc_index([pa, pr])

    # the accepted table's schema fallbacks are recorded, not silent
    accepted_key = next(key for key in fallbacks if key.endswith("accepted.csv"))
    reports_key = next(key for key in fallbacks if key.endswith("reports.csv"))
    assert fallbacks[accepted_key], "accepted-table reconciliation not reported"
    assert fallbacks[reports_key] == [], "canonical table should need no fallback"

    # accepted (OK) wins over reports (FAIL_LONG_BOND) for b_rescued
    assert qc_index.set_index("build_id").loc["b_rescued", "qc_class"] == "OK"

    ds = _dataset(["b_rescued", "b_plain", "b_fail", "b_noxyz", "b_absent"])
    ds, report = b.attach_geometry_status(ds, qc_index)

    got = dict(zip(ds["build_id"], ds["geometry_ok"]))
    # rescued+accepted -> clean; plain OK -> clean; long-bond fail -> not;
    # OK-but-no-xyz -> not; absent build_id -> not.
    assert got == {
        "b_rescued": True, "b_plain": True, "b_fail": False,
        "b_noxyz": False, "b_absent": False,
    }, got
    # rescued row must carry the accepted geometry's path, not the reports one
    assert ds.loc[ds["build_id"] == "b_rescued", "xyz_path"].iloc[0] == "regen/b_rescued.xyz"
    assert "schema_drift_warning" not in report


def test_schema_drift_warning_fires_when_xyz_gate_wipes_all_ok():
    # hand-built index: OK rows but xyz_exists all False -> the drift fingerprint
    qc_index = pd.DataFrame({
        "build_id": ["b1", "b2"], "qc_class": ["OK", "OK"],
        "xyz_exists": [False, False], "xyz_path": ["", ""], "mol2_path": ["", ""],
    })
    ds = _dataset(["b1", "b2"])
    ds, report = b.attach_geometry_status(ds, qc_index)
    assert report["clean_geometry_rows"] == 0
    assert "schema_drift_warning" in report


def test_qc_index_prefers_usable_ok_path_over_ok_without_path():
    ok_without_path = _write(pd.DataFrame({
        "build_id": ["b_shadow"],
        "qc_class": ["OK"],
        # Deliberately no xyz_path/xyz_exists columns: this must not outrank a
        # later clean row with an actual path.
        "coreCN_max_dist": [2.2],
    }), "ok_without_path.csv")
    ok_with_path = _write(pd.DataFrame({
        "build_id": ["b_shadow"],
        "qc_class": ["OK"],
        "xyz_exists": [True],
        "xyz_path": ["p/b_shadow.xyz"],
        "coreCN_max_dist": [2.4],
    }), "ok_with_path.csv")
    qc_index, *_ = b.load_geometry_qc_index([ok_without_path, ok_with_path])
    row = qc_index.set_index("build_id").loc["b_shadow"]
    assert row["xyz_path"] == "p/b_shadow.xyz"
    assert str(row["geometry_source"]).endswith("ok_with_path.csv")


def test_geometry_key_fallback_keeps_accepted_build_id_provenance():
    qc_index = pd.DataFrame({
        "build_id": ["accepted_old_plan"],
        "geometry_key": ["63|CCO|water"],
        "qc_class": ["OK"],
        "xyz_exists": [True],
        "xyz_path": ["accepted.xyz"],
        "mol2_path": ["accepted.mol2"],
        "geometry_source": ["accepted.csv"],
        "coreCN_max_dist": [2.5],
    })
    ds = _dataset(["newly_planned_id"])
    ds["geometry_key"] = "63|CCO|water"
    ds, report = b.attach_geometry_status(ds, qc_index)
    assert ds["geometry_ok"].tolist() == [True]
    assert ds["build_id"].tolist() == ["newly_planned_id"]
    assert ds["geometry_feature_build_id"].tolist() == ["accepted_old_plan"]
    assert report["rows_matched_by_geometry_key"] == 1


def test_qc_index_skips_empty_csv_and_records_it():
    empty = _REPO_ROOT / "tests" / "_tmp" / "empty.csv"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("\n", encoding="utf-8")
    pr = _write(_reports_table(), "nonempty.csv")
    qc_index, used, fallbacks = b.load_geometry_qc_index([empty, pr])
    empty_key = next(key for key in fallbacks if key.endswith("empty.csv"))
    assert fallbacks[empty_key] == ["skipped empty CSV"]
    assert len(qc_index) == len(_reports_table())
    assert str(empty) not in used


def test_numeric_looking_build_id_survives_csv_type_inference():
    numeric_id = _write(pd.DataFrame({
        "build_id": ["001234567890"],
        "qc_class": ["OK"],
        "xyz_exists": [True],
        "xyz_path": ["p/001234567890.xyz"],
    }), "numeric_build_id.csv")
    qc_index, *_ = b.load_geometry_qc_index([numeric_id])
    ds = _dataset(["001234567890"])
    ds, report = b.attach_geometry_status(ds, qc_index)

    assert ds["geometry_ok"].tolist() == [True]
    assert report["clean_geometry_rows"] == 1


def test_default_qc_tables_discover_isolated_family_acceptance_reports():
    old_reports_dir = b.REPORTS_DIR
    reports_dir = _REPO_ROOT / "tests" / "_tmp" / "reports"
    accepted = (
        reports_dir / "family_regeneration_runs" / "template_pincer"
        / "regenerated_fail_long_bond_accepted.csv"
    )
    accepted.parent.mkdir(parents=True, exist_ok=True)
    _accepted_table().to_csv(accepted, index=False)
    try:
        b.REPORTS_DIR = reports_dir
        tables = b.default_geometry_qc_tables()
    finally:
        b.REPORTS_DIR = old_reports_dir
    assert accepted in tables
    assert tables.index(accepted) < tables.index(reports_dir / "geometry_ok_for_features.csv")


def test_builtin_scalar_block_is_gated_on_geometry_ok():
    pr = _write(_reports_table(), "reports.csv")
    qc_index, *_ = b.load_geometry_qc_index([pr])
    ds = _dataset(["b_plain", "b_fail"])
    ds, _ = b.attach_geometry_status(ds, qc_index)
    scol = f"{b.FEATURE_3D_PREFIX}{b.BUILTIN_BLOCK_NAME}__coreCN_max_donor_dist"
    # clean row keeps the scalar; failed row is NaN
    assert ds[scol].notna().tolist() == [True, False]


def test_external_block_gated_vs_ungated():
    pr = _write(_reports_table(), "reports.csv")
    qc_index, *_ = b.load_geometry_qc_index([pr])
    ds = _dataset(["b_plain", "b_fail"])
    ds, _ = b.attach_geometry_status(ds, qc_index)

    blk = _write(pd.DataFrame({"build_id": ["b_plain", "b_fail"], "pi_0": [0.1, 0.9]}), "blk.csv")

    gated = b.parse_feature_block_specs([f"pig:{blk}"])
    ds, man_g = b.attach_feature_blocks(ds, gated)
    cg = f"{b.FEATURE_3D_PREFIX}pig__pi_0"
    assert ds[cg].notna().tolist() == [True, False], "gated block leaked into failed row"

    ungated = b.parse_feature_block_specs([f"piu:{blk}::ungated"])
    ds, man_u = b.attach_feature_blocks(ds, ungated)
    cu = f"{b.FEATURE_3D_PREFIX}piu__pi_0"
    assert ds[cu].notna().tolist() == [True, True], "ungated block wrongly gated"


def test_external_block_matched_rows_counts_any_feature_column():
    pr = _write(_reports_table(), "reports.csv")
    qc_index, *_ = b.load_geometry_qc_index([pr])
    ds = _dataset(["b_plain", "b_fail"])
    ds, _ = b.attach_geometry_status(ds, qc_index)

    blk = _write(pd.DataFrame({
        "build_id": ["b_plain", "b_fail"],
        "mostly_missing": [np.nan, np.nan],
        "real_value": [1.0, 2.0],
    }), "sparse_block.csv")
    gated = b.parse_feature_block_specs([f"sparse:{blk}"])
    ds, manifest = b.attach_feature_blocks(ds, gated)
    assert manifest[0]["matched_rows"] == 1

    ungated = b.parse_feature_block_specs([f"sparse_u:{blk}::ungated"])
    ds, manifest = b.attach_feature_blocks(ds, ungated)
    assert manifest[0]["matched_rows"] == 2


def test_sample_weights_normalised_to_mean_one():
    ds = _dataset(["b1"] * 6, metals=["La", "La", "La", "Eu", "Nd", "Nd"])
    ds, rep = b.compute_sample_weights(ds)
    wcol = f"{b.SAMPLE_WEIGHT_PREFIX}inv_metal_freq"
    assert abs(ds[wcol].mean() - 1.0) < 1e-9
    assert not b.is_feature_col(wcol), "weights must not be treated as features"


def test_final_columns_keep_meta_and_weights_but_not_as_features():
    pr = _write(_reports_table(), "reports.csv")
    qc_index, *_ = b.load_geometry_qc_index([pr])
    ds = _dataset(["b_plain"])
    ds[b.SMILES_COL] = "CCO"
    ds["D"] = 1.0
    ds[b.TARGET] = 0.0
    ds[b.GROUP_COL] = "CCO"
    ds[b.SPLIT_COL] = "unassigned"
    ds, _ = b.attach_geometry_status(ds, qc_index)
    wcol = f"{b.SAMPLE_WEIGHT_PREFIX}inv_metal_freq"
    ds[wcol] = 1.0
    final, _ = b.select_final_columns(ds, extra_keep=b.GEOMETRY_META_COLS + [wcol])
    assert "safe_exp_id" in final.columns and wcol in final.columns
    scol = f"{b.FEATURE_3D_PREFIX}{b.BUILTIN_BLOCK_NAME}__coreCN_max_donor_dist"
    assert b.is_feature_col(scol) and scol in final.columns


def test_safe_exp_id_is_namespaced_by_source_export():
    raw = pd.DataFrame({
        "Metal_Name": ["Eu", "La"],
        "obsDvaluesValue": [1.0, 2.0],
        "Extractant_SMILES": ["CCO", "CCO"],
        "Extractant_Name": ["ethanol", "ethanol"],
        "exp_id": ["123", "123"],
        "source_file": ["Eu_SAFE.csv", "La_SAFE.csv"],
        b.RAW_ROW_COL: [0, 1],
    })
    old_audit = b.AUDIT_DIR
    old_invalid = b.INVALID_SMILES_FILE
    tmp = _REPO_ROOT / "tests" / "_tmp" / "safe_id_audit"
    try:
        b.AUDIT_DIR = tmp
        b.INVALID_SMILES_FILE = tmp / "invalid.csv"
        clean, *_ = b.clean_and_dedup(raw)
    finally:
        b.AUDIT_DIR = old_audit
        b.INVALID_SMILES_FILE = old_invalid
    assert clean["safe_exp_id"].tolist() == ["Eu_SAFE:123", "La_SAFE:123"]
    assert clean["safe_exp_id"].is_unique


def test_existing_geometry_index_freezes_stage2_template_fields():
    key = "63|CCO|water"
    frame = pd.DataFrame({
        "geometry_key": [key], "build_id": ["new_plan"],
        "Atomic Number_metal": [63], "metal_symbol": ["Eu"], "metal_ox": [3],
        "SMILES_FOR_ARCHITECTOR": ["CCO"], "COORDLIST": ["[2]"],
        "DONOR_TYPES": ['["O(alcohol)"]'], "DENTATE": [1], "coreCN": [8],
        "n_ligs": [4], "inner_sphere_anion": ["water"],
        "fill_ligand": ["water"], "n_fill": [4],
        "geometry_status": ["planned"],
    })
    mapping = pd.DataFrame({"geometry_key": [key], "build_id": ["new_plan"]})
    frozen = frame.copy()
    frozen["build_id"] = "frozen_old"
    frozen["COORDLIST"] = "[1, 2]"
    frozen["DENTATE"] = 2
    frozen["coreCN"] = 9
    frozen["n_ligs"] = 3
    frozen["n_fill"] = 3
    index_path = _write(frozen, "frozen_index.csv")
    df_out, specs_out, map_out, report = b.freeze_plan_to_existing_geometry_index(
        frame, frame.copy(), mapping, {"coreCN_distribution": {8: 1}}, index_path
    )
    assert df_out.loc[0, "build_id"] == "frozen_old"
    assert specs_out.loc[0, "COORDLIST"] == "[1, 2]"
    assert specs_out.loc[0, "coreCN"] == 9
    assert map_out.loc[0, "build_id"] == "frozen_old"
    assert report["existing_geometry_plan_freeze"]["matched_specs"] == 1


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    # tidy up temp fixtures
    import shutil
    shutil.rmtree(_REPO_ROOT / "tests" / "_tmp", ignore_errors=True)
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
