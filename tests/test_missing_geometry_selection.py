"""Tests for selecting true missing canonical geometry outputs."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_builder():
    path = REPO_ROOT / "scripts" / "build_unique_geometries.py"
    spec = importlib.util.spec_from_file_location("build_unique_geometries_missing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b = _load_builder()

EXPECTED_RESCUE_BUILD_IDS = {
    "f3f070427c1f", "bd11f8c14383", "b72884625baf", "033b4c93146d",
    "a5b6774306ad", "e0bff762c05d", "0a5ee26d567d", "bef90f96562a",
    "e54b7e0eebc9", "7dc84a0bd047", "3927fc1f583b", "3dd657f759a8",
    "c25a798202a1", "c6b5c3f7cc44", "74b654cc8157", "b63ff46d0633",
    "68ff2a8a7c68", "9b52da38293d", "b7e4a26801b1", "5e4d792ef262",
    "9caebdec8648", "5e8f5789b5fd", "d655b942946b",
}


def _spec(build_id: str) -> dict:
    return {
        "build_id": build_id,
        "Atomic Number_metal": 63,
        "metal_symbol": "Eu",
        "metal_ox": 3,
        "coreCN": 9,
        "n_ligs": 3,
        "inner_sphere_anion": "nitrate",
    }


class MissingGeometrySelectionTests(unittest.TestCase):
    def test_uses_canonical_disk_state_not_recorded_status(self):
        with tempfile.TemporaryDirectory(prefix="missing_geometry_") as tmp:
            out = Path(tmp)
            specs = pd.DataFrame([_spec("present"), _spec("missing"), _spec("invalid")])

            present, _, _ = b.expected_paths(specs.iloc[0], out)
            present.parent.mkdir(parents=True, exist_ok=True)
            present.write_text("2\nvalid\nEu 0 0 0\nO 2 0 0\n", encoding="utf-8")

            invalid, _, _ = b.expected_paths(specs.iloc[2], out)
            invalid.write_text("not an xyz\n", encoding="utf-8")

            selected = b._missing_geometry_specs(specs, out)

            self.assertEqual(selected["build_id"].tolist(), ["missing", "invalid"])

    def test_valid_xyz_rejects_malformed_or_nonfinite_atom_rows(self):
        with tempfile.TemporaryDirectory(prefix="invalid_xyz_") as tmp:
            root = Path(tmp)
            malformed = root / "malformed.xyz"
            malformed.write_text("2\ncomment\nnot coordinates\nstill not coordinates\n", encoding="utf-8")
            nonfinite = root / "nonfinite.xyz"
            nonfinite.write_text("2\ncomment\nEu nan 0 0\nO 2 0 0\n", encoding="utf-8")

            self.assertFalse(b.valid_xyz(malformed))
            self.assertFalse(b.valid_xyz(nonfinite))

    def test_filter_before_sharding_assigns_exactly_one_of_23(self):
        specs = pd.DataFrame([_spec(f"missing_{i:02d}") for i in range(23)])
        assignments = [
            b._assigned_specs(specs, num_shards=23, shard_id=shard)
            for shard in range(23)
        ]
        self.assertTrue(all(len(part) == 1 for part in assignments))
        self.assertEqual(
            [part.iloc[0]["build_id"] for part in assignments],
            specs["build_id"].tolist(),
        )

    def test_every_operational_rescue_row_has_valid_explicit_ligtype(self):
        specs = pd.read_csv(REPO_ROOT / "data" / "processed" / "geometry_specs.csv", low_memory=False)
        rescue = specs[specs["build_id"].astype(str).isin(EXPECTED_RESCUE_BUILD_IDS)]
        self.assertEqual(set(rescue["build_id"].astype(str)), EXPECTED_RESCUE_BUILD_IDS)

        args = SimpleNamespace(ligtype_override_index=b.load_ligtype_overrides(
            REPO_ROOT / "data" / "processed" / "ligtype_overrides.csv"
        ))
        selected = [b._ligtype_override_for_spec(row, args) for _, row in rescue.iterrows()]
        self.assertEqual(len(selected), 23)
        self.assertTrue(all(selected))

    def test_attempt_seeds_are_distinct_across_profiles_and_retries(self):
        seeds = {
            b._attempt_seed(104729, 15485863, profile, attempt, 2)
            for profile in range(4)
            for attempt in (1, 2)
        }
        self.assertEqual(len(seeds), 8)

    def test_emergency_profile_uses_force_generation_without_relaxation(self):
        profile = b.BUILD_PROFILES["emergency_unrelaxed"]
        self.assertTrue(profile["force_generation"])
        self.assertFalse(profile["relax"])
        self.assertEqual(profile["full_method"], "UFF")

    def test_hard10_profiles_bypass_gfnff_preoptimization(self):
        relaxed = b.BUILD_PROFILES["uff_xtb_no_preopt"]
        unrelaxed = b.BUILD_PROFILES["uff_unrelaxed"]
        for profile in (relaxed, unrelaxed):
            self.assertFalse(profile["ff_preopt"])
            self.assertEqual(profile["assemble_method"], "UFF")
            self.assertTrue(profile["force_generation"])
        self.assertTrue(relaxed["relax"])
        self.assertEqual(relaxed["full_method"], "GFN2-xTB")
        self.assertFalse(unrelaxed["relax"])
        self.assertEqual(unrelaxed["full_method"], "UFF")

    def test_hard10_groups_are_two_disjoint_fives(self):
        no_structures = set(b.HARD10_NO_STRUCTURES_BUILD_IDS)
        native_crash = set(b.HARD10_NATIVE_CRASH_BUILD_IDS)
        self.assertEqual(len(no_structures), 5)
        self.assertEqual(len(native_crash), 5)
        self.assertFalse(no_structures & native_crash)

    def test_prepare_hard10_rescue_writes_ordered_isolated_queues(self):
        with tempfile.TemporaryDirectory(prefix="hard10_queues_") as tmp:
            tmp_path = Path(tmp)
            no_structures_path = tmp_path / "no_structures.csv"
            native_crash_path = tmp_path / "native_crash.csv"
            args = SimpleNamespace(
                specs=REPO_ROOT / "data" / "processed" / "geometry_specs.csv",
                ligtype_overrides=REPO_ROOT / "data" / "processed" / "ligtype_overrides.csv",
                hard10_no_structures_queue_output=no_structures_path,
                hard10_native_crash_queue_output=native_crash_path,
            )

            self.assertEqual(b.prepare_hard10_rescue(args), 0)
            no_structures = pd.read_csv(no_structures_path, low_memory=False)
            native_crash = pd.read_csv(native_crash_path, low_memory=False)

            self.assertEqual(
                no_structures["build_id"].astype(str).tolist(),
                list(b.HARD10_NO_STRUCTURES_BUILD_IDS),
            )
            self.assertEqual(
                native_crash["build_id"].astype(str).tolist(),
                list(b.HARD10_NATIVE_CRASH_BUILD_IDS),
            )
            self.assertTrue(no_structures["rescue_ligtype_override"].astype(str).str.len().gt(0).all())
            self.assertTrue(native_crash["rescue_ligtype_override"].astype(str).str.len().gt(0).all())

    def test_index_tag_is_part_of_shard_artifact_name(self):
        self.assertEqual(
            b._shard_artifact_name("geometry_index", 2, 5, "hard10_native_crash"),
            "geometry_index_hard10_native_crash_shard2of5.csv",
        )

    def test_dedup_prefers_preserved_failed_qc_xyz_over_old_no_xyz_failure(self):
        rows = []
        for status, xyz_path in [
            ("failed_native_crash_repeated", ""),
            ("failed_qc", "data/geometries/Eu/example.xyz"),
        ]:
            row = {field: "" for field in b.INDEX_FIELDS}
            row.update({"build_id": "rescued", "status": status, "xyz_path": xyz_path})
            rows.append(row)
        merged = b._dedup_best([pd.DataFrame(rows, columns=b.INDEX_FIELDS)])
        self.assertEqual(merged.iloc[0]["status"], "failed_qc")
        self.assertTrue(str(merged.iloc[0]["xyz_path"]).endswith("example.xyz"))


if __name__ == "__main__":
    unittest.main()
