"""Tests for partial regeneration report recovery."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_builder():
    path = REPO_ROOT / "scripts" / "build_unique_geometries.py"
    spec = importlib.util.spec_from_file_location("build_unique_geometries_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b = _load_builder()


def _args(
    tmp_path: Path,
    queue: Path,
    *,
    allow_missing: bool,
    num_shards: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        reports_dir=tmp_path,
        num_shards=num_shards,
        allow_missing_shard_reports=allow_missing,
        regenerate_input=queue,
        specs=queue,
        long_bond_threshold=3.10,
        ambiguous_gap_threshold=0.10,
        borderline_longish_threshold=2.95,
        regen_out=tmp_path / "geometries",
        max_attempts=2,
        timeout_per_complex=1800,
        seed=0xF00D,
        seed_step=7919,
        profile_sequence="standard",
        n_symmetries=40,
        n_symmetries_step=10,
        n_conformers=5,
        n_conformers_step=1,
        xtb_max_iterations=250,
        ligtype_overrides=tmp_path / "missing_ligtype_overrides.csv",
        run_id="",
        queue_sha256="",
    )


def _spec_row(build_id: str) -> dict:
    return {
        "build_id": build_id,
        "qc_class": "FAIL_LONG_BOND",
        "Atomic Number_metal": 63,
        "metal_symbol": "Eu",
        "metal_ox": 3,
        "SMILES_FOR_ARCHITECTOR": "O",
        "smiles_for_architector_used": "O",
        "COORDLIST": "[0]",
        "DENTATE": 1,
        "coreCN": 1,
        "n_ligs": 1,
        "inner_sphere_anion": "water",
        "fill_ligand": "water",
        "n_fill": 0,
        "geometry_key": f"63|O|water|{build_id}",
    }


def _accepted_xyz(tmp_path: Path, build_id: str) -> Path:
    path = tmp_path / "geometries" / "accepted" / "Eu" / f"Eu_{build_id}.xyz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("2\naccepted\nEu 0 0 0\nO 2.3 0 0\n", encoding="utf-8")
    return path


def _write_shard(tmp_path: Path, name: str, shard_id: int, rows: list[dict]) -> None:
    stem = Path(name).stem
    suffix = Path(name).suffix
    pd.DataFrame(rows).to_csv(tmp_path / f"{stem}_shard{shard_id}of2{suffix}", index=False)


class RegenerationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix="regen_recovery_")
        self.tmp_path = Path(self._tempdir.name)

    def tearDown(self):
        self._tempdir.cleanup()

    def _queue(self) -> Path:
        queue = self.tmp_path / "queue.csv"
        pd.DataFrame([_spec_row("b1"), _spec_row("b2")]).to_csv(queue, index=False)
        return queue

    def test_partial_merge_writes_summary_and_returns_incomplete(self):
        queue = self._queue()
        accepted_xyz = _accepted_xyz(self.tmp_path, "b1")
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ATTEMPTS_NAME,
            0,
            [{"queue_index": 1, "build_id": "b1", "attempt": 1, "qc_class": "OK"}],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ACCEPTED_NAME,
            0,
            [{"queue_index": 1, "build_id": "b1", "qc_class": "OK",
              "accepted_xyz_path": str(accepted_xyz)}],
        )

        result = b.merge_regenerated(_args(self.tmp_path, queue, allow_missing=True))
        self.assertEqual(result, 2)
        summary = (self.tmp_path / b.REGENERATE_SUMMARY_NAME).read_text(encoding="utf-8")
        self.assertIn("Rows completed: 1", summary)
        self.assertIn("Rows incomplete: 1", summary)
        self.assertIn("Missing shard reports: 1", summary)
        self.assertIn("WARNING: regeneration is incomplete", summary)

    def test_complete_merge_returns_success(self):
        queue = self._queue()
        accepted_xyz = _accepted_xyz(self.tmp_path, "b1")
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ATTEMPTS_NAME,
            0,
            [{"queue_index": 1, "build_id": "b1", "attempt": 1, "qc_class": "OK"}],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ATTEMPTS_NAME,
            1,
            [{"queue_index": 2, "build_id": "b2", "attempt": 1, "qc_class": "FAIL_LONG_BOND"}],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ACCEPTED_NAME,
            0,
            [{"queue_index": 1, "build_id": "b1", "qc_class": "OK",
              "accepted_xyz_path": str(accepted_xyz)}],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_STILL_FAILED_NAME,
            1,
            [{"queue_index": 2, "build_id": "b2", "best_qc_class": "FAIL_LONG_BOND"}],
        )

        result = b.merge_regenerated(_args(self.tmp_path, queue, allow_missing=True))
        self.assertEqual(result, 0)
        summary = (self.tmp_path / b.REGENERATE_SUMMARY_NAME).read_text(encoding="utf-8")
        self.assertIn("Rows completed: 2", summary)
        self.assertIn("Rows incomplete: 0", summary)
        self.assertIn("Missing shard reports: 0", summary)

    def test_single_shard_merge_reads_canonical_reports_and_repairs_empty_csv(self):
        queue = self.tmp_path / "single_queue.csv"
        pd.DataFrame({
            "build_id": ["b1"],
            "qc_class": ["FAIL_LONG_BOND"],
        }).to_csv(queue, index=False)
        pd.DataFrame([{
            "queue_index": 1,
            "build_id": "b1",
            "attempt": 1,
            "qc_class": "FAIL_LONG_BOND",
        }]).to_csv(self.tmp_path / b.REGENERATE_ATTEMPTS_NAME, index=False)
        # Reproduce the one-byte file emitted by an older zero-acceptance merge.
        (self.tmp_path / b.REGENERATE_ACCEPTED_NAME).write_text("\n", encoding="utf-8")
        pd.DataFrame([{
            "queue_index": 1,
            "build_id": "b1",
            "best_qc_class": "FAIL_LONG_BOND",
        }]).to_csv(self.tmp_path / b.REGENERATE_STILL_FAILED_NAME, index=False)

        result = b.merge_regenerated(
            _args(self.tmp_path, queue, allow_missing=False, num_shards=1)
        )

        self.assertEqual(result, 0)
        failed = pd.read_csv(self.tmp_path / b.REGENERATE_STILL_FAILED_NAME)
        accepted = pd.read_csv(self.tmp_path / b.REGENERATE_ACCEPTED_NAME)
        self.assertEqual(failed["build_id"].tolist(), ["b1"])
        self.assertEqual(len(accepted), 0)
        self.assertIn("build_id", accepted.columns)
        summary = (self.tmp_path / b.REGENERATE_SUMMARY_NAME).read_text(encoding="utf-8")
        self.assertIn("Rows completed: 1", summary)
        self.assertIn("Rows incomplete: 0", summary)

    def test_merge_ignores_stale_rows_and_remaps_current_queue(self):
        queue = self._queue()
        accepted_xyz = _accepted_xyz(self.tmp_path, "b1")
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ATTEMPTS_NAME,
            0,
            [
                {"queue_index": 3, "build_id": "obsolete", "attempt": 1, "qc_class": "FAIL_LONG_BOND"},
                {"queue_index": 9, "build_id": "b1", "attempt": 1, "qc_class": "OK"},
            ],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ATTEMPTS_NAME,
            1,
            [{"queue_index": 2, "build_id": "b2", "attempt": 1, "qc_class": "FAIL_LONG_BOND"}],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ACCEPTED_NAME,
            0,
            [{
                "queue_index": 9,
                "build_id": "b1",
                "qc_class": "OK",
                "accepted_for_clean_3d_features": True,
                "accepted_xyz_path": str(accepted_xyz),
            }],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_STILL_FAILED_NAME,
            0,
            [{"queue_index": 8, "build_id": "b2", "best_qc_class": "FAIL_LONG_BOND"}],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_STILL_FAILED_NAME,
            1,
            [{"queue_index": 2, "build_id": "b2", "best_qc_class": "FAIL_LONG_BOND"}],
        )

        result = b.merge_regenerated(_args(self.tmp_path, queue, allow_missing=True))

        self.assertEqual(result, 0)
        attempts = pd.read_csv(self.tmp_path / b.REGENERATE_ATTEMPTS_NAME)
        accepted = pd.read_csv(self.tmp_path / b.REGENERATE_ACCEPTED_NAME)
        failed = pd.read_csv(self.tmp_path / b.REGENERATE_STILL_FAILED_NAME)
        self.assertEqual(set(attempts["build_id"]), {"b1", "b2"})
        self.assertEqual(accepted[["queue_index", "build_id"]].values.tolist(), [[1, "b1"]])
        self.assertEqual(failed[["queue_index", "build_id"]].values.tolist(), [[2, "b2"]])
        summary = (self.tmp_path / b.REGENERATE_SUMMARY_NAME).read_text(encoding="utf-8")
        self.assertIn("Rows completed: 2", summary)
        self.assertIn("Rows incomplete: 0", summary)

    def test_merge_preserves_acceptance_from_different_pilot_cardinality(self):
        queue = self._queue()
        accepted_xyz = _accepted_xyz(self.tmp_path, "b1")
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ATTEMPTS_NAME,
            0,
            [{"queue_index": 2, "build_id": "b2", "attempt": 1,
              "qc_class": "BORDERLINE_AMBIGUOUS_SHELL"}],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_ATTEMPTS_NAME,
            1,
            [{"queue_index": 2, "build_id": "b2", "attempt": 2,
              "qc_class": "BORDERLINE_AMBIGUOUS_SHELL"}],
        )
        _write_shard(
            self.tmp_path,
            b.REGENERATE_STILL_FAILED_NAME,
            0,
            [{"queue_index": 2, "build_id": "b2",
              "best_qc_class": "BORDERLINE_AMBIGUOUS_SHELL"}],
        )
        pilot_name = Path(b.REGENERATE_ACCEPTED_NAME)
        pd.DataFrame([{
            "queue_index": 1,
            "build_id": "b1",
            "qc_class": "OK",
            "accepted_for_clean_3d_features": True,
            "accepted_xyz_path": str(accepted_xyz),
        }]).to_csv(
            self.tmp_path / f"{pilot_name.stem}_shard0of8{pilot_name.suffix}",
            index=False,
        )

        result = b.merge_regenerated(_args(self.tmp_path, queue, allow_missing=True))

        self.assertEqual(result, 0)
        accepted = pd.read_csv(self.tmp_path / b.REGENERATE_ACCEPTED_NAME)
        failed = pd.read_csv(self.tmp_path / b.REGENERATE_STILL_FAILED_NAME)
        self.assertEqual(accepted[["queue_index", "build_id"]].values.tolist(), [[1, "b1"]])
        self.assertEqual(failed[["queue_index", "build_id"]].values.tolist(), [[2, "b2"]])
        summary = (self.tmp_path / b.REGENERATE_SUMMARY_NAME).read_text(encoding="utf-8")
        self.assertIn("Rows completed: 2", summary)
        self.assertIn("Rows incomplete: 0", summary)

    def test_repeated_ambiguous_shell_stops_only_non_long_bond_retries(self):
        attempts = [
            {"qc_class": "BORDERLINE_AMBIGUOUS_SHELL", "coreCN_max_dist": 2.60},
            {"qc_class": "BORDERLINE_AMBIGUOUS_SHELL", "coreCN_max_dist": 2.55},
        ]
        self.assertTrue(b._repeated_ambiguous_shell(attempts, 3.10))
        attempts[-1]["coreCN_max_dist"] = 3.20
        self.assertFalse(b._repeated_ambiguous_shell(attempts, 3.10))
        attempts[-1] = {"qc_class": "FAIL_LONG_BOND", "coreCN_max_dist": 3.20}
        self.assertFalse(b._repeated_ambiguous_shell(attempts, 3.10))

    def test_dual_qc_rejects_nearest_ok_when_required_nitrate_is_missing(self):
        xyz = self.tmp_path / "nitrate_replaced_by_water.xyz"
        xyz.write_text(
            "3\nsynthetic\nEu 0 0 0\nO 2.20 0 0\nO 0 2.30 0\n",
            encoding="utf-8",
        )
        spec = {
            **_spec_row("nitrate_missing"),
            "coreCN": 2,
            "inner_sphere_anion": "nitrate",
            "fill_ligand": "nitrate",
            "n_fill": 1,
        }
        nearest = b.nearest_corecn_qc_xyz(xyz, spec)
        result = b._dual_qc_candidate(
            xyz,
            spec,
            SimpleNamespace(
                long_bond_threshold=3.10,
                borderline_longish_threshold=2.95,
                ambiguous_gap_threshold=0.10,
            ),
        )

        self.assertEqual(nearest["qc_class"], "OK")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["file_qc_status"], "NITRATE_MISSING")

    def test_file_qc_requires_every_implied_nitrate_to_be_inner_sphere(self):
        xyz = self.tmp_path / "one_of_two_nitrates_remote.xyz"
        atoms = [
            ("Eu", 0.0, 0.0, 0.0),
            ("O", -2.0, 0.0, 0.0),  # ligand oxygen
            ("N", 0.0, 0.0, 2.5),
            ("O", 0.0, 0.0, 1.3),
            ("O", 1.2, 0.0, 2.5),
            ("O", -1.2, 0.0, 2.5),
            ("N", 10.0, 0.0, 0.0),
            ("O", 8.8, 0.0, 0.0),
            ("O", 10.0, 1.2, 0.0),
            ("O", 10.0, -1.2, 0.0),
        ]
        xyz.write_text(
            f"{len(atoms)}\nsynthetic\n" + "".join(
                f"{symbol} {x:.3f} {y:.3f} {z:.3f}\n"
                for symbol, x, y, z in atoms
            ),
            encoding="utf-8",
        )
        spec = {
            **_spec_row("two_nitrates"),
            "coreCN": 5,
            "inner_sphere_anion": "nitrate",
            "fill_ligand": "nitrate",
            "n_fill": 4,
        }

        status, note = b.qc_xyz(xyz, pd.Series(spec))

        self.assertEqual(status, "NITRATE_MISSING")
        self.assertIn("inner_sphere_nitrate_count=1;expected=2", note)

    def test_ensure_csv_atomically_upgrades_an_old_report_header(self):
        report = self.tmp_path / "old_attempts.csv"
        pd.DataFrame([{
            "queue_index": 1,
            "build_id": "b1",
            "attempt": 1,
            "qc_class": "BUILD_FAILED",
        }]).to_csv(report, index=False)

        with mock.patch.object(b, "_write_csv_atomic", wraps=b._write_csv_atomic) as atomic_write:
            b._ensure_csv(report, b.REGEN_ATTEMPT_FIELDS)

        atomic_write.assert_called_once()
        with report.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(next(csv.reader(handle)), b.REGEN_ATTEMPT_FIELDS)
        upgraded = pd.read_csv(report)
        self.assertEqual(upgraded.loc[0, "build_id"], "b1")
        self.assertEqual(int(upgraded.loc[0, "attempt"]), 1)
        self.assertFalse(list(self.tmp_path.glob(f".{report.name}.*.tmp")))

    def test_run_meta_fingerprint_includes_the_expected_shard(self):
        queue = self._queue()
        args = _args(self.tmp_path, queue, allow_missing=False, num_shards=2)
        args.run_id = "immutable-run"
        args.queue_sha256 = b._sha256_file(queue)
        args.strategy_sha256 = b._regeneration_strategy_sha256(args)
        args.shard_id = 0
        meta = self.tmp_path / "meta.json"
        b._write_regeneration_run_meta(meta, args)

        self.assertTrue(b._run_meta_matches(meta, args, expected_shard_id=0))
        self.assertFalse(b._run_meta_matches(meta, args, expected_shard_id=1))
        args.queue_sha256 = "wrong"
        self.assertFalse(b._run_meta_matches(meta, args, expected_shard_id=0))

    def test_strict_run_resumes_after_last_recorded_attempt(self):
        queue = self.tmp_path / "queue.csv"
        specs = self.tmp_path / "specs.csv"
        pd.DataFrame([_spec_row("b1")]).to_csv(queue, index=False)
        pd.DataFrame([_spec_row("b1")]).to_csv(specs, index=False)
        reports = self.tmp_path / "reports"
        reports.mkdir()
        args = _args(reports, queue, allow_missing=False, num_shards=1)
        args.specs = specs
        args.regen_out = self.tmp_path / "regenerated"
        args.run_id = "resume-run"
        args.shard_id = 0
        args.max_attempts = 3
        args.overwrite_accepted = False
        args.limit = None
        args.queue_sha256 = b._sha256_file(queue)
        args.strategy_sha256 = b._regeneration_strategy_sha256(args)

        prior = {
            "queue_index": 1,
            "build_id": "b1",
            "run_id": args.run_id,
            "queue_sha256": args.queue_sha256,
            "strategy_sha256": args.strategy_sha256,
            "attempt": 1,
            "attempt_status": "build_failed",
            "qc_class": "BUILD_FAILED",
            "qc_note": "interrupted_after_attempt_record",
        }
        b._append_csv_row(
            reports / b.REGENERATE_ATTEMPTS_NAME,
            prior,
            b.REGEN_ATTEMPT_FIELDS,
        )
        b._write_regeneration_run_meta(
            reports / b.REGENERATE_RUN_META_NAME,
            args,
        )

        def failed_attempt(queue_index, spec, source_xyz, attempt, profile, out_root, call_args):
            return {
                "queue_index": queue_index,
                "build_id": str(spec["build_id"]),
                "run_id": call_args.run_id,
                "queue_sha256": call_args.queue_sha256,
                "strategy_sha256": call_args.strategy_sha256,
                "attempt": attempt,
                "profile": profile,
                "attempt_status": "build_failed",
                "accepted_for_clean_3d_features": False,
                "generated_xyz_path": "",
                "generated_mol2_path": "",
                "qc_class": "BUILD_FAILED",
                "qc_note": f"failed_attempt_{attempt}",
                "note": f"failed_attempt_{attempt}",
            }

        with mock.patch.object(
            b, "_run_regeneration_attempt", side_effect=failed_attempt,
        ) as run_attempt:
            self.assertEqual(b.regenerate_failed(args), 0)

        self.assertEqual(
            [call.args[3] for call in run_attempt.call_args_list],
            [2, 3],
        )
        recorded = pd.read_csv(reports / b.REGENERATE_ATTEMPTS_NAME)
        self.assertEqual(recorded["attempt"].astype(int).tolist(), [1, 2, 3])

    def test_strict_worker_refuses_to_overwrite_mismatched_run_meta(self):
        queue = self._queue()
        reports = self.tmp_path / "strict_reports"
        reports.mkdir()
        args = _args(reports, queue, allow_missing=False, num_shards=1)
        args.regen_out = self.tmp_path / "strict_geometries"
        args.run_id = "immutable-run"
        args.shard_id = 0
        args.queue_sha256 = b._sha256_file(queue)
        args.strategy_sha256 = b._regeneration_strategy_sha256(args)
        b._write_regeneration_run_meta(
            reports / b.REGENERATE_RUN_META_NAME,
            args,
        )

        args.seed += 1
        with self.assertRaisesRegex(SystemExit, "Immutable regeneration run metadata mismatch"):
            b.regenerate_failed(args)

    def test_strict_merge_rejects_meta_copied_from_another_shard(self):
        queue = self._queue()
        args = _args(self.tmp_path, queue, allow_missing=False, num_shards=2)
        args.run_id = "strict-merge"
        args.queue_sha256 = b._sha256_file(queue)
        args.strategy_sha256 = b._regeneration_strategy_sha256(args)
        args.shard_id = 0
        meta_paths = b._regen_shard_report_paths(
            self.tmp_path, b.REGENERATE_RUN_META_NAME, 2,
        )
        b._write_regeneration_run_meta(meta_paths[0], args)
        meta_paths[1].write_text(
            meta_paths[0].read_text(encoding="utf-8"), encoding="utf-8",
        )

        self.assertEqual(b.merge_regenerated(args), 1)

    def test_resume_materializes_pilot_acceptance_in_current_shard(self):
        reports = self.tmp_path / "reports"
        out_root = self.tmp_path / "geometries"
        accepted_xyz = out_root / "accepted" / "Eu" / "Eu_b1.xyz"
        accepted_xyz.parent.mkdir(parents=True)
        accepted_xyz.write_text("2\naccepted\nEu 0 0 0\nO 2.3 0 0\n", encoding="utf-8")

        row = _spec_row("b1")
        queue = self.tmp_path / "queue.csv"
        specs = self.tmp_path / "specs.csv"
        pd.DataFrame([row]).to_csv(queue, index=False)
        pd.DataFrame([row]).to_csv(specs, index=False)
        reports.mkdir()
        pilot_name = Path(b.REGENERATE_ACCEPTED_NAME)
        pd.DataFrame([{
            "queue_index": 1,
            "build_id": "b1",
            "qc_class": "OK",
            "accepted_for_clean_3d_features": True,
            "accepted_xyz_path": str(accepted_xyz),
        }]).to_csv(
            reports / f"{pilot_name.stem}_shard0of8{pilot_name.suffix}",
            index=False,
        )
        args = SimpleNamespace(
            reports_dir=reports,
            regen_out=out_root,
            regenerate_input=queue,
            specs=specs,
            num_shards=2,
            shard_id=0,
            overwrite_accepted=False,
            limit=None,
            profile_sequence="standard",
            ligtype_overrides=self.tmp_path / "missing_overrides.csv",
            max_attempts=3,
            timeout_per_complex=1800,
            seed=0xF00D,
            seed_step=7919,
            n_symmetries=40,
            n_symmetries_step=10,
            n_conformers=5,
            n_conformers_step=1,
            xtb_max_iterations=250,
            long_bond_threshold=3.10,
            ambiguous_gap_threshold=0.10,
            borderline_longish_threshold=2.95,
            run_id="",
            queue_sha256="",
        )

        self.assertEqual(b.regenerate_failed(args), 0)
        current = pd.read_csv(
            reports / "regenerated_fail_long_bond_accepted_shard0of2.csv"
        )
        self.assertEqual(current[["queue_index", "build_id"]].values.tolist(), [[1, "b1"]])
        self.assertEqual(
            pd.read_csv(reports / "regenerated_fail_long_bond_attempts_shard0of2.csv").shape[0],
            0,
        )

    def test_prepare_missing_regeneration_selects_only_rows_without_xyz(self):
        queue = self.tmp_path / "all_fail_long_bond.csv"
        pd.DataFrame({
            "build_id": ["missing", "has_candidate", "already_ok"],
            "qc_class": ["FAIL_LONG_BOND", "FAIL_LONG_BOND", "OK"],
            "metal_symbol": ["Eu", "Eu", "Eu"],
        }).to_csv(queue, index=False)
        still_failed = self.tmp_path / "still_failed.csv"
        pd.DataFrame({
            "build_id": ["missing", "has_candidate"],
            "best_xyz_path": ["", "candidate.xyz"],
            "failure_note": ["timeout", "coreCN_max_dist>3.10"],
            "accepted_for_clean_3d_features": [False, False],
        }).to_csv(still_failed, index=False)
        output = self.tmp_path / "rescue.csv"
        args = SimpleNamespace(
            regenerate_input=queue,
            still_failed_input=still_failed,
            rescue_queue_output=output,
        )

        result = b.prepare_missing_regeneration(args)

        self.assertEqual(result, 0)
        rescue = pd.read_csv(output)
        self.assertEqual(rescue["build_id"].tolist(), ["missing"])
        self.assertEqual(rescue["rescue_previous_failure"].tolist(), ["timeout"])
        self.assertEqual(rescue["rescue_reason"].tolist(), ["regeneration_produced_no_xyz"])


if __name__ == "__main__":
    unittest.main()
