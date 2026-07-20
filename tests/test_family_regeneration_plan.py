"""Tests for non-overlapping chemistry-family regeneration planning."""

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

from src.chemistry.coordination import complex_build_id  # noqa: E402


def _load_builder():
    path = REPO_ROOT / "scripts" / "build_unique_geometries.py"
    spec = importlib.util.spec_from_file_location("build_unique_geometries_family", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b = _load_builder()


def _queue_row(build_id: str, smiles: str, coord: list[int], denticity: int) -> dict:
    return {
        "build_id": build_id,
        "qc_class": "FAIL_LONG_BOND",
        "Atomic Number_metal": 63,
        "metal_symbol": "Eu",
        "metal_ox": 3,
        "SMILES_FOR_ARCHITECTOR": smiles,
        "smiles_for_architector_used": smiles,
        "COORDLIST": json.dumps(coord),
        "DONOR_TYPES": "[]",
        "DENTATE": denticity,
        "coreCN": 9,
        "n_ligs": max(1, 9 // denticity),
        "inner_sphere_anion": "nitrate",
        "fill_ligand": "nitrate",
        "n_fill": max(0, 9 - max(1, 9 // denticity) * denticity),
        "geometry_key": f"63|{smiles}|nitrate",
    }


class FamilyRegenerationPlanTests(unittest.TestCase):
    def test_routes_each_source_once_and_replans_btp_with_new_build_id(self):
        btp = "CCCc1nnc(-c2cccc(-c3nnc(CCC)c(CCC)n3)n2)nc1CCC"
        dga = "CN(C)C(=O)COCC(=O)N(C)C"
        simple_chelate = "O=C(O)CC(=O)O"
        pytri = "CCCCCCCCn1cc(-c2cc(CC(=O)O)cc(-c3cn(CCCCCCCC)nn3)n2)nn1"
        dga_donors = b.detect_donors(dga)
        chelate_donors = b.detect_donors(simple_chelate)
        pytri_donors = b.detect_donors(pytri)
        self.assertIsNotNone(dga_donors)
        self.assertIsNotNone(chelate_donors)
        self.assertIsNotNone(pytri_donors)

        btp_row = _queue_row("old_btp", btp, [4, 5, 13, 14, 23, 24, 25], 7)
        btp_row.update({
            "Atomic Number_metal": 70,
            "metal_symbol": "Yb",
            "coreCN": 8,
            "n_ligs": 1,
            "n_fill": 1,
            "geometry_key": f"70|{btp}|nitrate",
        })
        queue = pd.DataFrame([
            btp_row,
            _queue_row("old_pytri", pytri, [8, 22, 31, 32, 33, 34, 35], 7),
            _queue_row("borderline", dga, dga_donors.coord_list, dga_donors.denticity),
            _queue_row("no_xyz", simple_chelate, chelate_donors.coord_list, chelate_donors.denticity),
            _queue_row("manual", "CCCC", [0], 1),
        ])
        failed = pd.DataFrame([
            {"build_id": "old_btp", "best_coreCN_max_dist": 4.0,
             "best_xyz_path": "old.xyz", "failure_note": "long"},
            {"build_id": "old_pytri", "best_coreCN_max_dist": 4.0,
             "best_xyz_path": "old_pytri.xyz", "failure_note": "long"},
            {"build_id": "borderline", "best_coreCN_max_dist": 3.2,
             "best_xyz_path": "borderline.xyz", "failure_note": "long"},
            {"build_id": "no_xyz", "best_coreCN_max_dist": "",
             "best_xyz_path": "", "failure_note": "no_structures_returned"},
            {"build_id": "manual", "best_coreCN_max_dist": 4.0,
             "best_xyz_path": "manual.xyz", "failure_note": "long"},
        ])

        with tempfile.TemporaryDirectory(prefix="family_regen_") as tmp:
            root = Path(tmp)
            queue_path = root / "queue.csv"
            failed_path = root / "failed.csv"
            out = root / "plan"
            queue.to_csv(queue_path, index=False)
            failed.to_csv(failed_path, index=False)
            args = SimpleNamespace(
                regenerate_input=queue_path,
                still_failed_input=failed_path,
                family_plan_dir=out,
                reports_dir=root,
                specs=root / "absent_specs.csv",
            )

            self.assertEqual(b.prepare_family_regeneration(args), 0)

            replan = pd.read_csv(out / "template_replan.csv")
            borderline = pd.read_csv(out / "conformer_borderline.csv")
            no_xyz = pd.read_csv(out / "no_xyz_uff.csv")
            manual = pd.read_csv(out / "manual_review.csv")
            self.assertEqual(replan["source_build_id"].tolist(), ["old_btp", "old_pytri"])
            self.assertNotEqual(replan.iloc[0]["build_id"], "old_btp")
            self.assertEqual(int(replan.iloc[0]["DENTATE"]), 3)
            self.assertEqual(json.loads(replan.iloc[0]["COORDLIST"]), [5, 13, 24])
            self.assertEqual(int(replan.iloc[0]["coreCN"]), 9)
            self.assertEqual(int(replan.iloc[0]["n_ligs"]), 3)
            self.assertEqual(int(replan.iloc[0]["n_fill"]), 0)
            expected_id = complex_build_id(
                metal_Z=70,
                ligand_smiles=btp,
                coord_list=[5, 13, 24],
                denticity=3,
                core_cn=9,
                n_ligs=3,
                inner_sphere_anion="nitrate",
                fill_ligand="nitrate",
                n_fill_value=0,
            )
            self.assertEqual(replan.iloc[0]["build_id"], expected_id)
            self.assertEqual(json.loads(replan.iloc[1]["COORDLIST"]), [32, 33, 34])
            self.assertEqual(int(replan.iloc[1]["coreCN"]), 9)
            self.assertEqual(int(replan.iloc[1]["n_ligs"]), 3)
            self.assertEqual(int(replan.iloc[1]["n_fill"]), 0)
            self.assertEqual(borderline["source_build_id"].tolist(), ["borderline"])
            self.assertEqual(no_xyz["source_build_id"].tolist(), ["no_xyz"])
            self.assertEqual(manual["source_build_id"].tolist(), ["manual"])

            routed = []
            for name in (
                "template_replan", "conformer_borderline", "conformer_sibling",
                "conformer_deep", "no_xyz_uff", "no_xyz_ligtype", "manual_review",
            ):
                frame = pd.read_csv(out / f"{name}.csv")
                routed.extend(frame.get("source_build_id", pd.Series(dtype=str)).astype(str))
            self.assertEqual(len(routed), 5)
            self.assertEqual(
                set(routed), {"old_btp", "old_pytri", "borderline", "no_xyz", "manual"},
            )

            overrides = pd.read_csv(out / "template_ligtype_overrides.csv")
            self.assertEqual(overrides["ligType"].tolist(), ["tri_mer_bent", "tri_mer_bent"])

            status_args = SimpleNamespace(
                family_plan_dir=out,
                family_runs_dir=root / "runs",
            )
            self.assertEqual(b.family_regeneration_status(status_args), 0)
            status = pd.read_csv(out / "status.csv")
            self.assertEqual(int(status["planned"].sum()), 4)
            self.assertEqual(int(status["pending"].sum()), 4)
            self.assertTrue(status.loc[status["planned"] > 0, "state"].eq("pending").all())

    def test_prepare_adaptive_regeneration_uses_fill_parity_not_family_name(self):
        smiles = "n1ccccc1-c2ccccn2"
        rows = [
            {
                "build_id": "cn9_ambiguous",
                "best_qc_class": "BORDERLINE_AMBIGUOUS_SHELL",
                "attempts_run": 2,
                "best_coreCN_max_dist": 2.61,
                "best_gap_after_coreCN": 0.03,
                "best_xyz_path": "candidate.xyz",
                "Atomic Number_metal": 63,
                "metal_symbol": "Eu",
                "metal_ox": 3,
                "smiles_for_architector_used": smiles,
                "COORDLIST": json.dumps([0, 7, 8, 13]),
                "DENTATE": 4,
                "coreCN": 9,
                "n_ligs": 1,
                "inner_sphere_anion": "nitrate",
                "fill_ligand": "nitrate",
                "geometry_key": f"63|{smiles}|nitrate",
            },
            {
                "build_id": "cn8_already_even",
                "best_qc_class": "BORDERLINE_AMBIGUOUS_SHELL",
                "attempts_run": 2,
                "best_coreCN_max_dist": 2.61,
                "Atomic Number_metal": 63,
                "metal_symbol": "Eu",
                "metal_ox": 3,
                "smiles_for_architector_used": smiles,
                "COORDLIST": json.dumps([0, 7, 8, 13]),
                "DENTATE": 4,
                "coreCN": 8,
                "n_ligs": 1,
                "inner_sphere_anion": "nitrate",
                "fill_ligand": "nitrate",
            },
            {
                "build_id": "not_diagnostic",
                "best_qc_class": "FAIL_LONG_BOND",
                "attempts_run": 2,
                "best_coreCN_max_dist": 3.50,
                "Atomic Number_metal": 63,
                "metal_symbol": "Eu",
                "metal_ox": 3,
                "smiles_for_architector_used": smiles,
                "COORDLIST": json.dumps([0, 7, 8, 13]),
                "DENTATE": 4,
                "coreCN": 9,
                "n_ligs": 1,
                "inner_sphere_anion": "nitrate",
                "fill_ligand": "nitrate",
            },
        ]

        with tempfile.TemporaryDirectory(prefix="adaptive_regen_") as tmp:
            root = Path(tmp)
            input_path = root / "still_failed.csv"
            attempts_path = root / b.REGENERATE_ATTEMPTS_NAME
            output_path = root / "adaptive.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)
            pd.DataFrame([
                {"build_id": build_id, "attempt": attempt,
                 "qc_class": "BORDERLINE_AMBIGUOUS_SHELL"}
                for build_id in ("cn9_ambiguous", "cn8_already_even")
                for attempt in (1, 2)
            ]).to_csv(attempts_path, index=False)
            args = SimpleNamespace(
                adaptive_input=input_path,
                adaptive_attempts_input=attempts_path,
                adaptive_output=output_path,
                long_bond_threshold=3.10,
            )

            self.assertEqual(b.prepare_adaptive_regeneration(args), 0)
            adaptive = pd.read_csv(output_path)
            self.assertEqual(adaptive["source_build_id"].tolist(), ["cn9_ambiguous"])
            self.assertEqual(int(adaptive.iloc[0]["original_coreCN"]), 9)
            self.assertEqual(int(adaptive.iloc[0]["original_secondary_water_count"]), 1)
            self.assertEqual(int(adaptive.iloc[0]["coreCN"]), 8)
            self.assertEqual(int(adaptive.iloc[0]["n_fill"]), 4)
            self.assertEqual(int(adaptive.iloc[0]["proposed_nitrate_count"]), 2)
            self.assertEqual(int(adaptive.iloc[0]["proposed_secondary_water_count"]), 0)
            self.assertEqual(adaptive.iloc[0]["rescue_route"], "adaptive_cn_fill")
            expected_id = complex_build_id(
                metal_Z=63,
                ligand_smiles=smiles,
                coord_list=[0, 7, 8, 13],
                denticity=4,
                core_cn=8,
                n_ligs=1,
                inner_sphere_anion="nitrate",
                fill_ligand="nitrate",
                n_fill_value=4,
            )
            self.assertEqual(adaptive.iloc[0]["build_id"], expected_id)
            self.assertEqual(adaptive.iloc[0]["source_xyz_path"], "candidate.xyz")

    def test_prepare_adaptive_requires_two_actual_ambiguous_attempt_rows(self):
        with tempfile.TemporaryDirectory(prefix="adaptive_actual_attempts_") as tmp:
            root = Path(tmp)
            input_path = root / "still_failed.csv"
            attempts_path = root / "attempts.csv"
            output_path = root / "adaptive.csv"
            row = _queue_row("aggregate_only", "n1ccccc1", [0], 1)
            row.update({
                "best_qc_class": "BORDERLINE_AMBIGUOUS_SHELL",
                "attempts_run": 9,
                "ambiguous_attempts": 9,
                "best_coreCN_max_dist": 2.60,
                "n_ligs": 2,
                "coreCN": 9,
                "n_fill": 7,
            })
            pd.DataFrame([row]).to_csv(input_path, index=False)
            duplicate_attempt = {
                "build_id": "aggregate_only",
                "attempt": 1,
                "profile": "standard",
                "seed": 123,
                "qc_class": "BORDERLINE_AMBIGUOUS_SHELL",
            }
            # Canonical and shard reports can both contain this same physical
            # attempt; that is still one diagnosis, not two independent tries.
            pd.DataFrame([duplicate_attempt, duplicate_attempt]).to_csv(
                attempts_path, index=False,
            )

            args = SimpleNamespace(
                adaptive_input=input_path,
                adaptive_attempts_input=attempts_path,
                adaptive_output=output_path,
                long_bond_threshold=3.10,
            )
            self.assertEqual(b.prepare_adaptive_regeneration(args), 0)
            self.assertEqual(len(pd.read_csv(output_path)), 0)

    def test_family_status_reconciles_pilot_and_full_shard_cardinalities(self):
        with tempfile.TemporaryDirectory(prefix="family_status_") as tmp:
            root = Path(tmp)
            plan = root / "plan"
            report_dir = root / "runs" / "group"
            plan.mkdir()
            report_dir.mkdir(parents=True)
            pd.DataFrame([{
                "run_group": "group",
                "queue": str(plan / "group.csv"),
                "rows": 2,
            }]).to_csv(plan / "run_config.csv", index=False)
            pd.DataFrame({"build_id": ["pilot_ok", "full_failed"]}).to_csv(
                plan / "group.csv", index=False,
            )
            pd.DataFrame([{
                "build_id": "pilot_ok",
                "qc_class": "OK",
                "accepted_for_clean_3d_features": True,
            }]).to_csv(
                report_dir / "regenerated_fail_long_bond_accepted_shard0of8.csv",
                index=False,
            )
            pd.DataFrame([{
                "build_id": "full_failed",
                "best_qc_class": "BORDERLINE_AMBIGUOUS_SHELL",
            }]).to_csv(
                report_dir / "regenerated_fail_long_bond_still_failed_shard0of16.csv",
                index=False,
            )

            args = SimpleNamespace(family_plan_dir=plan, family_runs_dir=root / "runs")
            self.assertEqual(b.family_regeneration_status(args), 0)
            status = pd.read_csv(plan / "status.csv").iloc[0]
            self.assertEqual(int(status["accepted"]), 1)
            self.assertEqual(int(status["still_failed"]), 1)
            self.assertEqual(int(status["pending"]), 0)
            self.assertEqual(status["state"], "complete")

    def test_family_status_accepts_headerless_zero_acceptance_report(self):
        with tempfile.TemporaryDirectory(prefix="family_status_empty_") as tmp:
            root = Path(tmp)
            plan = root / "plan"
            report_dir = root / "runs" / "group"
            plan.mkdir()
            report_dir.mkdir(parents=True)
            pd.DataFrame([{
                "run_group": "group",
                "queue": str(plan / "group.csv"),
                "rows": 1,
            }]).to_csv(plan / "run_config.csv", index=False)
            pd.DataFrame({"build_id": ["failed"]}).to_csv(
                plan / "group.csv", index=False,
            )
            (report_dir / b.REGENERATE_ACCEPTED_NAME).write_text("\n", encoding="utf-8")
            pd.DataFrame([{
                "build_id": "failed",
                "best_qc_class": "FAIL_LONG_BOND",
            }]).to_csv(
                report_dir / b.REGENERATE_STILL_FAILED_NAME,
                index=False,
            )

            args = SimpleNamespace(family_plan_dir=plan, family_runs_dir=root / "runs")
            self.assertEqual(b.family_regeneration_status(args), 0)
            status = pd.read_csv(plan / "status.csv").iloc[0]
            self.assertEqual(int(status["accepted"]), 0)
            self.assertEqual(int(status["still_failed"]), 1)
            self.assertEqual(int(status["pending"]), 0)
            self.assertEqual(status["state"], "complete")


if __name__ == "__main__":
    unittest.main()
