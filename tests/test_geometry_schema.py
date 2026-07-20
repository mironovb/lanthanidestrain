"""Contract tests for the shared geometry QC schema.

These guard the exact failure mode that produced the silent stage-E bug: the
producer (build_unique_geometries.py) and the consumer (build_dataset_no3d.py)
drifting apart on CSV column names. If someone renames a column on one side, one
of these fails loudly instead of the pipeline shipping a wrong geometry_ok.

Runnable with pytest, or standalone: ``.venv/bin/python tests/test_geometry_schema.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import geometry_schema as g  # noqa: E402


def _load_producer():
    """Import scripts/build_unique_geometries.py by path (it is not a package)."""
    path = _REPO_ROOT / "scripts" / "build_unique_geometries.py"
    spec = importlib.util.spec_from_file_location("build_unique_geometries", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_producer_emits_the_aliases_the_consumer_normalizes():
    """Every alias the consumer knows how to reconcile must actually be written
    by the producer's accepted-table field list -- otherwise the normaliser is
    reconciling a column that no longer exists."""
    producer = _load_producer()
    accepted_fields = set(producer.REGEN_ACCEPTED_FIELDS)
    assert set(g.PATH_ALIASES).issubset(accepted_fields), (
        f"producer no longer emits {set(g.PATH_ALIASES) - accepted_fields}; "
        f"schema PATH_ALIASES has drifted from REGEN_ACCEPTED_FIELDS"
    )
    for canonical in (g.BUILD_ID, g.QC_CLASS, g.CORECN_MAX_DIST, g.GAP_AFTER_CORECN):
        assert canonical in accepted_fields, f"producer dropped {canonical!r}"


def test_normalize_accepted_shape_table():
    """Accepted table: accepted_* paths, no xyz_exists -> canonical + derived."""
    table = pd.DataFrame({
        g.BUILD_ID: ["b1"],
        g.QC_CLASS: ["OK"],
        g.ACCEPTED_XYZ_PATH: ["regen/b1.xyz"],
        g.ACCEPTED_MOL2_PATH: ["regen/b1.mol2"],
    })
    out, applied = g.normalize_qc_table(table)
    assert g.XYZ_PATH in out.columns and g.MOL2_PATH in out.columns
    assert g.XYZ_EXISTS in out.columns and bool(out[g.XYZ_EXISTS].iloc[0]) is True
    assert applied, "fallbacks should be reported, not silent"


def test_normalize_canonical_table_is_a_noop():
    """Reports table already in canonical shape -> no fallbacks reported."""
    table = pd.DataFrame({
        g.BUILD_ID: ["b2"],
        g.QC_CLASS: ["OK"],
        g.XYZ_EXISTS: [True],
        g.XYZ_PATH: ["p/b2.xyz"],
        g.MOL2_PATH: ["p/b2.mol2"],
    })
    out, applied = g.normalize_qc_table(table)
    assert applied == []
    assert out[g.XYZ_EXISTS].iloc[0] == True  # noqa: E712  (keep as-is)


def test_normalize_derives_false_for_empty_path():
    table = pd.DataFrame({g.BUILD_ID: ["b3"], g.QC_CLASS: ["OK"], g.XYZ_PATH: [""]})
    out, applied = g.normalize_qc_table(table)
    assert bool(out[g.XYZ_EXISTS].iloc[0]) is False
    assert any("derived" in a for a in applied)


def test_normalize_without_path_defaults_not_usable():
    table = pd.DataFrame({g.BUILD_ID: ["b4"], g.QC_CLASS: ["OK"]})
    out, applied = g.normalize_qc_table(table)
    assert bool(out[g.XYZ_EXISTS].iloc[0]) is False
    assert any("False" in a for a in applied)


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
