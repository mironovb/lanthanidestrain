"""Focused tests for the bounded all-remaining regeneration plan."""

from __future__ import annotations

import importlib.util
import json
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
    spec = importlib.util.spec_from_file_location("build_unique_geometries_all_remaining", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b = _load_builder()


def _spec(build_id: str, *, nitrate_parity: bool = False) -> dict:
    if nitrate_parity:
        return {
            "build_id": build_id,
            "Atomic Number_metal": 63,
            "metal_symbol": "Eu",
            "metal_ox": 3,
            "SMILES_FOR_ARCHITECTOR": "NCCNCCNCCN",
            "COORDLIST": json.dumps([0, 3, 6, 9]),
            "DONOR_TYPES": json.dumps(["N", "N", "N", "N"]),
            "DENTATE": 4,
            "coreCN": 9,
            "n_ligs": 2,
            "inner_sphere_anion": "nitrate",
            "fill_ligand": "nitrate",
            "n_fill": 1,
            "geometry_key": f"63|{build_id}|nitrate",
        }
    return {
        "build_id": build_id,
        "Atomic Number_metal": 63,
        "metal_symbol": "Eu",
        "metal_ox": 3,
        "SMILES_FOR_ARCHITECTOR": "O",
        "COORDLIST": "[0]",
        "DONOR_TYPES": '["O"]',
        "DENTATE": 1,
        "coreCN": 1,
        "n_ligs": 1,
        "inner_sphere_anion": "water",
        "fill_ligand": "water",
        "n_fill": 0,
        "geometry_key": f"63|{build_id}|water",
    }


def _planner_args(root: Path, *, remaining_scope: str) -> SimpleNamespace:
    return SimpleNamespace(
        specs=root / "specs.csv",
        geometry_index=root / "index.csv",
        ligtype_overrides=root / "missing_ligtype_overrides.csv",
        family_plan_dir=root / "missing_family_plan",
        family_runs_dir=root / "missing_family_runs",
        adaptive_output=root / "missing_adaptive.csv",
        historical_reports_dir=root / "missing_historical_reports",
        all_remaining_plan_dir=root / "plan",
        all_remaining_runs_dir=root / "runs",
        all_remaining_out_dir=root / "geometries",
        hypothesis_version="test-v1",
        remaining_scope=remaining_scope,
        long_bond_threshold=3.10,
        borderline_longish_threshold=2.95,
        ambiguous_gap_threshold=0.10,
    )


class AllRemainingRegenerationTests(unittest.TestCase):
    def test_aminopoly_cn8_uses_dual_qc_family_precedent_across_ligands(self):
        with tempfile.TemporaryDirectory(prefix="all_remaining_aminopoly_precedent_") as tmp:
            root = Path(tmp)
            family_plan = root / "family_plan"
            family_reports = root / "family_runs" / "template_aminopolycarboxylate"
            family_plan.mkdir()
            family_reports.mkdir(parents=True)
            precedent_smiles = "NCCNCCNCCNCCNCCN"
            target_smiles = "NCCNCCNCCNCCNCCNC"

            def aminopoly_row(build_id: str, root_id: str, smiles: str, core_cn: int) -> dict:
                return {
                    "build_id": build_id,
                    "source_build_id": root_id,
                    "root_source_build_id": root_id,
                    "parent_build_id": root_id,
                    "Atomic Number_metal": 63,
                    "metal_symbol": "Eu",
                    "metal_ox": 3,
                    "SMILES_FOR_ARCHITECTOR": smiles,
                    "smiles_for_architector_used": smiles,
                    "COORDLIST": json.dumps([0, 3, 6, 9, 12, 15]),
                    "DONOR_TYPES": json.dumps(["N"] * 6),
                    "DENTATE": 6,
                    "coreCN": core_cn,
                    "n_ligs": 1,
                    "inner_sphere_anion": "nitrate",
                    "fill_ligand": "nitrate",
                    "n_fill": core_cn - 6,
                    "geometry_key": f"63|{root_id}|nitrate",
                }

            precedent = aminopoly_row("precedent_cn8", "precedent_root", precedent_smiles, 8)
            target = aminopoly_row("target_cn9", "target_root", target_smiles, 9)
            baseline_precedent = {**precedent, "build_id": "precedent_root"}
            baseline_target = {**target, "build_id": "target_root"}
            pd.DataFrame([baseline_precedent, baseline_target]).to_csv(
                root / "specs.csv", index=False,
            )
            pd.DataFrame(columns=["build_id", "status", "xyz_path", "qc_status"]).to_csv(
                root / "index.csv", index=False,
            )
            family_queue = family_plan / "template_aminopolycarboxylate.csv"
            pd.DataFrame([precedent, target]).to_csv(family_queue, index=False)
            pd.DataFrame([{
                "run_group": "template_aminopolycarboxylate",
                "queue": str(family_queue),
                "rows": 2,
            }]).to_csv(family_plan / "run_config.csv", index=False)

            xyz = root / "precedent_cn8.xyz"
            ligand_n = [
                ("N", 2.00, 0.00, 0.00), ("N", -2.05, 0.00, 0.00),
                ("N", 0.00, 2.10, 0.00), ("N", 0.00, -2.15, 0.00),
                ("N", 0.00, 0.00, -2.20), ("N", 1.30, 1.30, 1.00),
            ]
            carbons = [("C", 5.0 + i, 5.0, 5.0) for i in range(10)]
            nitrate = [
                ("N", 0.00, 0.00, 2.60),
                ("O", 0.00, 0.00, 1.40),
                ("O", 1.50, 0.00, 2.60),
                ("O", -1.50, 0.00, 2.60),
            ]
            atoms = [("Eu", 0.0, 0.0, 0.0), *ligand_n, *carbons, *nitrate]
            xyz.write_text(
                f"{len(atoms)}\nsynthetic family precedent\n" + "".join(
                    f"{symbol} {x:.3f} {y:.3f} {z:.3f}\n"
                    for symbol, x, y, z in atoms
                ),
                encoding="utf-8",
            )
            self.assertEqual(b.qc_xyz(xyz, pd.Series(precedent))[0], "accepted")
            self.assertEqual(b.nearest_corecn_qc_xyz(xyz, precedent)["qc_class"], "OK")
            pd.DataFrame([{
                "build_id": "precedent_cn8",
                "accepted_xyz_path": str(xyz),
                "qc_class": "OK",
                "accepted_for_clean_3d_features": True,
            }]).to_csv(family_reports / b.REGENERATE_ACCEPTED_NAME, index=False)
            pd.DataFrame([{
                "build_id": "target_cn9",
                "best_qc_class": "BUILD_FAILED",
                "best_xyz_path": "",
                "failure_note": "no_structures_returned",
            }]).to_csv(family_reports / b.REGENERATE_STILL_FAILED_NAME, index=False)

            args = _planner_args(root, remaining_scope="known-unfinished")
            args.family_plan_dir = family_plan
            args.family_runs_dir = root / "family_runs"
            self.assertEqual(b.prepare_all_remaining(args), 0)
            manifest = pd.read_csv(root / "plan" / "manifest.csv")
            self.assertEqual(manifest["root_source_build_id"].tolist(), ["target_root"])
            self.assertEqual(manifest["route"].tolist(), ["aminopoly_cn8"])
            queue = pd.read_csv(root / "plan" / "aminopoly_cn8.csv")
            self.assertEqual(int(queue.iloc[0]["coreCN"]), 8)
            self.assertEqual(int(queue.iloc[0]["n_fill"]), 2)
            self.assertEqual(
                json.loads(queue.iloc[0]["family_precedent_build_ids"]),
                ["precedent_cn8"],
            )

    def test_family_candidate_reported_accepted_is_requeued_when_file_qc_fails(self):
        with tempfile.TemporaryDirectory(prefix="all_remaining_family_reaudit_") as tmp:
            root = Path(tmp)
            family_plan = root / "family_plan"
            family_reports = root / "family_runs" / "conformer_deep"
            family_plan.mkdir()
            family_reports.mkdir(parents=True)
            baseline = _spec("family_root")
            candidate = {
                **baseline,
                "build_id": "family_candidate",
                "source_build_id": "family_root",
                "root_source_build_id": "family_root",
                "parent_build_id": "family_root",
                "coreCN": 2,
                "inner_sphere_anion": "nitrate",
                "fill_ligand": "nitrate",
                "n_fill": 1,
            }
            pd.DataFrame([baseline]).to_csv(root / "specs.csv", index=False)
            pd.DataFrame(columns=["build_id", "status", "xyz_path", "qc_status"]).to_csv(
                root / "index.csv", index=False,
            )
            family_queue = family_plan / "conformer_deep.csv"
            pd.DataFrame([candidate]).to_csv(family_queue, index=False)
            pd.DataFrame([{
                "run_group": "conformer_deep",
                "queue": str(family_queue),
                "rows": 1,
            }]).to_csv(family_plan / "run_config.csv", index=False)
            xyz = root / "nearest_ok_but_no_nitrate.xyz"
            xyz.write_text(
                "3\nsynthetic\nEu 0 0 0\nO 2.20 0 0\nO 0 2.30 0\n",
                encoding="utf-8",
            )
            pd.DataFrame([{
                "build_id": "family_candidate",
                "accepted_xyz_path": str(xyz),
                "qc_class": "OK",
                "accepted_for_clean_3d_features": True,
            }]).to_csv(family_reports / b.REGENERATE_ACCEPTED_NAME, index=False)
            pd.DataFrame([{
                "build_id": "family_candidate",
                "best_qc_class": "NITRATE_MISSING",
                "best_xyz_path": str(xyz),
                "failure_note": "current dual QC rejected historical acceptance",
            }]).to_csv(family_reports / b.REGENERATE_STILL_FAILED_NAME, index=False)

            args = _planner_args(root, remaining_scope="known-unfinished")
            args.family_plan_dir = family_plan
            args.family_runs_dir = root / "family_runs"
            self.assertEqual(b.prepare_all_remaining(args), 0)
            manifest = pd.read_csv(root / "plan" / "manifest.csv")
            self.assertEqual(manifest["state"].tolist(), ["queued"])
            self.assertEqual(manifest["route"].tolist(), ["placement_qc"])
            self.assertEqual(manifest["root_source_build_id"].tolist(), ["family_root"])

    def test_known_unfinished_scope_omits_success_and_routes_nitrate_parity(self):
        with tempfile.TemporaryDirectory(prefix="all_remaining_scope_") as tmp:
            root = Path(tmp)
            pd.DataFrame([
                _spec("already_ok"),
                _spec("nitrate_failed", nitrate_parity=True),
            ]).to_csv(root / "specs.csv", index=False)
            pd.DataFrame([
                {"build_id": "already_ok", "status": "ok",
                 "xyz_path": "/cluster/path/not_mounted_here.xyz", "qc_status": "accepted"},
                {"build_id": "nitrate_failed", "status": "failed_qc",
                 "xyz_path": "", "qc_status": "NITRATE_MISSING"},
            ]).to_csv(root / "index.csv", index=False)

            args = _planner_args(root, remaining_scope="known-unfinished")
            self.assertEqual(b.prepare_all_remaining(args), 0)

            manifest = pd.read_csv(root / "plan" / "manifest.csv")
            self.assertEqual(manifest["root_source_build_id"].tolist(), ["nitrate_failed"])
            self.assertEqual(manifest["route"].tolist(), ["nitrate_parity_cn"])
            queue = pd.read_csv(root / "plan" / "nitrate_parity_cn.csv")
            self.assertEqual(len(queue), 1)
            self.assertEqual(int(queue.iloc[0]["coreCN"]), 8)
            self.assertEqual(int(queue.iloc[0]["n_fill"]), 0)
            self.assertEqual(queue.iloc[0]["parent_build_id"], "nitrate_failed")
            config = pd.read_csv(root / "plan" / "run_config.csv")
            nitrate = config[config["run_group"] == "nitrate_parity_cn"].iloc[0]
            self.assertEqual(int(nitrate["rows"]), 1)
            self.assertEqual(int(nitrate["num_shards"]), 1)
            self.assertEqual(int(nitrate["max_concurrent_tasks"]), 1)
            summary = (root / "plan" / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Scope: known-unfinished", summary)
            self.assertIn("Deferred existing successes: 1", summary)
            self.assertIn("Targeted sources in manifest: 1", summary)

    def test_strict_baseline_audit_keeps_unavailable_success_visible(self):
        with tempfile.TemporaryDirectory(prefix="all_remaining_strict_scope_") as tmp:
            root = Path(tmp)
            pd.DataFrame([_spec("already_ok")]).to_csv(root / "specs.csv", index=False)
            pd.DataFrame([{
                "build_id": "already_ok",
                "status": "ok",
                "xyz_path": "/cluster/path/not_mounted_here.xyz",
                "qc_status": "accepted",
            }]).to_csv(root / "index.csv", index=False)

            args = _planner_args(root, remaining_scope="strict-baseline-audit")
            self.assertEqual(b.prepare_all_remaining(args), 2)
            manifest = pd.read_csv(root / "plan" / "manifest.csv")
            self.assertEqual(manifest["state"].tolist(), ["blocked_unavailable"])
            self.assertEqual(manifest["root_source_build_id"].tolist(), ["already_ok"])

    def test_empty_target_manifest_preserves_baseline_coverage_and_status_is_ok(self):
        with tempfile.TemporaryDirectory(prefix="all_remaining_empty_manifest_") as tmp:
            root = Path(tmp)
            pd.DataFrame([_spec("already_ok")]).to_csv(root / "specs.csv", index=False)
            pd.DataFrame([{
                "build_id": "already_ok",
                "status": "ok",
                "xyz_path": "/cluster/path/not_needed_by_targeted_scope.xyz",
                "qc_status": "accepted",
            }]).to_csv(root / "index.csv", index=False)
            planner_args = _planner_args(root, remaining_scope="known-unfinished")
            self.assertEqual(b.prepare_all_remaining(planner_args), 0)
            self.assertEqual(len(pd.read_csv(root / "plan" / "manifest.csv")), 0)

            status_args = SimpleNamespace(
                all_remaining_plan_dir=root / "plan",
                all_remaining_runs_dir=root / "status",
                long_bond_threshold=3.10,
                borderline_longish_threshold=2.95,
                ambiguous_gap_threshold=0.10,
            )
            self.assertEqual(b.all_remaining_status(status_args), 0)
            status = pd.read_csv(root / "status" / "all_remaining_status.csv")
            self.assertEqual(len(status), 0)
            summary = (root / "status" / "all_remaining_status_summary.txt").read_text(
                encoding="utf-8",
            )
            self.assertIn("Baseline sources: 1", summary)
            self.assertIn("Deferred existing successes: 1", summary)

    def test_resolved_existing_becomes_invalid_if_geometry_disappears(self):
        with tempfile.TemporaryDirectory(prefix="all_remaining_resolved_drift_") as tmp:
            root = Path(tmp)
            pd.DataFrame([_spec("historically_resolved")]).to_csv(
                root / "specs.csv", index=False,
            )
            xyz = root / "resolved.xyz"
            xyz.write_text(
                "2\nsynthetic\nEu 0 0 0\nO 2.30 0 0\n",
                encoding="utf-8",
            )
            pd.DataFrame([{
                "build_id": "historically_resolved",
                "status": "failed_qc",
                "xyz_path": str(xyz),
                "qc_status": "accepted",
            }]).to_csv(root / "index.csv", index=False)
            planner_args = _planner_args(root, remaining_scope="known-unfinished")
            self.assertEqual(b.prepare_all_remaining(planner_args), 0)
            manifest = pd.read_csv(root / "plan" / "manifest.csv")
            self.assertEqual(manifest["state"].tolist(), ["resolved_existing"])
            self.assertTrue(str(manifest.iloc[0]["resolution_spec_json"]).strip())

            xyz.unlink()
            status_args = SimpleNamespace(
                all_remaining_plan_dir=root / "plan",
                all_remaining_runs_dir=root / "status",
                long_bond_threshold=3.10,
                borderline_longish_threshold=2.95,
                ambiguous_gap_threshold=0.10,
            )
            self.assertEqual(b.all_remaining_status(status_args), 2)
            status = pd.read_csv(root / "status" / "all_remaining_status.csv")
            self.assertEqual(status["final_state"].tolist(), ["invalid_resolved_existing"])

    def test_final_status_requires_matching_strict_meta_and_reports_rejection(self):
        with tempfile.TemporaryDirectory(prefix="all_remaining_status_") as tmp:
            root = Path(tmp)
            plan = root / "plan"
            reports = root / "reports" / "canonical_missing"
            plan.mkdir()
            reports.mkdir(parents=True)
            queue_path = plan / "canonical_missing.csv"
            pd.DataFrame([_spec("planned")]).to_csv(queue_path, index=False)
            queue_hash = b._sha256_file(queue_path)
            run_id = "test-v1:canonical_missing:strict"
            strategy = "strategy-sha"
            pd.DataFrame([{
                "root_source_build_id": "root",
                "baseline_build_id": "root",
                "state": "queued",
                "route": "canonical_missing",
                "planned_build_id": "planned",
                "parent_build_id": "root",
            }]).to_csv(plan / "manifest.csv", index=False)
            pd.DataFrame([{
                "run_group": "canonical_missing",
                "queue": str(queue_path),
                "rows": 1,
                "num_shards": 1,
                "run_id": run_id,
                "queue_sha256": queue_hash,
                "reports_dir": str(reports),
            }]).to_csv(plan / "run_config.csv", index=False)
            (plan / b.ALL_REMAINING_PLAN_META_NAME).write_text(json.dumps({
                "baseline_count": 1,
                "deferred_existing_successes": 0,
                "manifest_count": 1,
                "queued_count": 1,
                "manifest_sha256": b._sha256_file(plan / "manifest.csv"),
                "run_config_sha256": b._sha256_file(plan / "run_config.csv"),
            }), encoding="utf-8")
            (reports / b.REGENERATE_RUN_META_NAME).write_text(json.dumps({
                "run_id": run_id,
                "queue_sha256": queue_hash,
                "strategy_sha256": strategy,
                "num_shards": 1,
                "shard_id": 0,
            }), encoding="utf-8")
            pd.DataFrame([{
                "build_id": "planned",
                "run_id": run_id,
                "queue_sha256": queue_hash,
                "strategy_sha256": strategy,
                "best_qc_class": "BUILD_FAILED",
            }]).to_csv(reports / b.REGENERATE_STILL_FAILED_NAME, index=False)

            args = SimpleNamespace(
                all_remaining_plan_dir=plan,
                all_remaining_runs_dir=root / "status",
                long_bond_threshold=3.10,
                borderline_longish_threshold=2.95,
                ambiguous_gap_threshold=0.10,
            )
            self.assertEqual(b.all_remaining_status(args), 3)
            status = pd.read_csv(root / "status" / "all_remaining_status.csv")
            self.assertEqual(status["final_state"].tolist(), ["scientifically_rejected"])

            original_queue = queue_path.read_bytes()
            queue_path.write_bytes(original_queue + b"\n")
            self.assertEqual(b.all_remaining_status(args), 2)
            status = pd.read_csv(root / "status" / "all_remaining_status.csv")
            self.assertEqual(status["final_state"].tolist(), ["integrity_error"])
            queue_path.write_bytes(original_queue)

            meta = json.loads((reports / b.REGENERATE_RUN_META_NAME).read_text(encoding="utf-8"))
            meta["strategy_sha256"] = "new-strategy-without-a-merge"
            (reports / b.REGENERATE_RUN_META_NAME).write_text(
                json.dumps(meta), encoding="utf-8",
            )
            self.assertEqual(b.all_remaining_status(args), 2)
            status = pd.read_csv(root / "status" / "all_remaining_status.csv")
            self.assertEqual(status["final_state"].tolist(), ["pending_or_missing_report"])


if __name__ == "__main__":
    unittest.main()
