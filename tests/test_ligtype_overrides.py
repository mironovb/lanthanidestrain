"""Tests for manual Architector ligType override plumbing.

Runnable with pytest, or standalone:
``.venv/bin/python tests/test_ligtype_overrides.py``.
"""

from __future__ import annotations

import importlib.util
import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_builder():
    path = _REPO_ROOT / "scripts" / "build_unique_geometries.py"
    spec = importlib.util.spec_from_file_location("build_unique_geometries", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b = _load_builder()
_TMPDIR = tempfile.TemporaryDirectory(prefix="ligtype_overrides_")


def _write(df: pd.DataFrame, name: str) -> Path:
    out = Path(_TMPDIR.name)
    path = out / name
    df.to_csv(path, index=False)
    return path


def _spec(**overrides) -> pd.Series:
    base = {
        "build_id": "b1",
        "geometry_key": "g1",
        "SMILES_FOR_ARCHITECTOR": "O=C(O)CC(=O)O",
        "COORDLIST": "[0, 4]",
        "DENTATE": 2,
    }
    base.update(overrides)
    return pd.Series(base)


def test_build_id_override_is_selected_and_validated():
    path = _write(pd.DataFrame({
        "build_id": ["b1"],
        "ligType": ["bi_cis_chelating"],
        "note": ["manual O,O chelate"],
    }), "ligtype_override_build_id.csv")
    args = SimpleNamespace(ligtype_override_index=b.load_ligtype_overrides(path))
    assert b._ligtype_override_for_spec(_spec(), args) == "bi_cis_chelating"


def test_smiles_coord_override_uses_coordlist_key_not_smiles_only():
    path = _write(pd.DataFrame({
        "smiles_for_architector_used": ["O=C(O)CC(=O)O"],
        "COORDLIST": ["[0,4]"],
        "ligType": ["bi_cis"],
    }), "ligtype_override_smiles_coord.csv")
    args = SimpleNamespace(ligtype_override_index=b.load_ligtype_overrides(path))
    assert b._ligtype_override_for_spec(_spec(build_id="absent", geometry_key="absent"), args) == "bi_cis"
    assert b._ligtype_override_for_spec(
        _spec(build_id="absent2", geometry_key="absent2", COORDLIST="[1, 4]"), args
    ) == ""


def test_denticity_mismatch_fails_loudly():
    path = _write(pd.DataFrame({"build_id": ["b1"], "ligType": ["tri_mer"]}), "ligtype_bad_dent.csv")
    args = SimpleNamespace(ligtype_override_index=b.load_ligtype_overrides(path))
    try:
        b._ligtype_override_for_spec(_spec(), args)
    except SystemExit as exc:
        assert "denticity=3" in str(exc)
    else:
        raise AssertionError("denticity mismatch should fail")


def test_skipped_known_bad_is_explicitly_retryable_not_retry_failed_default():
    assert "skipped_known_bad_ligtype" in b.RETRYABLE_STATUSES
    retry_args = SimpleNamespace(retry_failed=True, retry_status=None)
    assert "skipped_known_bad_ligtype" not in b._resolve_retry_statuses(retry_args)
    explicit_args = SimpleNamespace(retry_failed=False, retry_status=["skipped_known_bad_ligtype"])
    assert "skipped_known_bad_ligtype" in b._resolve_retry_statuses(explicit_args)


def test_simplified_ligand_is_legacy_retryable_not_success():
    assert "ok_simplified_ligand" not in b.KEEP_STATUSES
    assert "ok_simplified_ligand" not in b.SUCCESS_STATUSES
    assert "ok_simplified_ligand" in b.RETRYABLE_STATUSES
    retry_args = SimpleNamespace(retry_failed=True, retry_status=None)
    assert "ok_simplified_ligand" not in b._resolve_retry_statuses(retry_args)
    explicit_args = SimpleNamespace(retry_failed=False, retry_status=["ok_simplified_ligand"])
    assert "ok_simplified_ligand" in b._resolve_retry_statuses(explicit_args)


def test_recovered_existing_does_not_hide_legacy_simplified_ligand():
    rows = []
    for status, note, simplified in [
        ("ok_simplified_ligand", "original_ligtype_failed_simplified_success", True),
        ("existing_ok", "recovered_from_disk", False),
    ]:
        row = {field: "" for field in b.INDEX_FIELDS}
        row.update({
            "build_id": "legacy_simplified",
            "status": status,
            "note": note,
            "xyz_path": "data/geometries/x.xyz",
            "simplified_ligand": simplified,
        })
        rows.append(row)

    deduped = b._dedup_best([pd.DataFrame(rows, columns=b.INDEX_FIELDS)])
    assert len(deduped) == 1
    assert deduped.iloc[0]["status"] == "ok_simplified_ligand"
    assert deduped.iloc[0]["note"] == "original_ligtype_failed_simplified_success"
    assert str(deduped.iloc[0]["simplified_ligand"]).lower() == "true"


def test_legacy_simplified_ids_force_rebuild_on_retry_statuses():
    recorded = {
        "old_failed": "failed_ligtype",
        "old_skipped": "skipped_known_bad_ligtype",
        "old_simplified": "ok_simplified_ligand",
        "real_ok": "ok",
    }
    retry_statuses = {"failed_ligtype", "skipped_known_bad_ligtype", "ok_simplified_ligand"}
    legacy_simplified_ids = {"old_failed", "old_skipped", "old_simplified", "real_ok"}

    force = b._force_rebuild_build_ids(recorded, retry_statuses, legacy_simplified_ids)
    assert force == {"old_failed", "old_skipped", "old_simplified"}


def test_mixed_schema_index_csv_is_read_and_upgraded():
    path = Path(_TMPDIR.name) / "mixed_schema_index.csv"
    old_fields = b.INDEX_FIELDS[:-1]
    old_row = {field: "" for field in old_fields}
    old_row.update({"build_id": "old", "status": "skipped_known_bad_ligtype"})
    new_row = {field: "" for field in b.INDEX_FIELDS}
    new_row.update({"build_id": "new", "status": "failed_ligtype", "ligtype_override": "tri_mer"})

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow(old_row)
        handle.write(",".join(new_row[field] for field in b.INDEX_FIELDS) + "\n")

    frame = b.read_index_csv(path)
    assert list(frame["build_id"]) == ["old", "new"]
    assert list(frame["ligtype_override"]) == ["", "tri_mer"]

    b.upgrade_index_csv_schema(path)
    upgraded = b.read_index_csv(path)
    assert list(upgraded.columns) == b.INDEX_FIELDS
    assert list(upgraded["ligtype_override"]) == ["", "tri_mer"]


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
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
