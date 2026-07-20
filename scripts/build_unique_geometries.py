#!/usr/bin/env python3
"""Build one 3D geometry (.xyz + .mol2) per unique complex -- failure-safe.

Stage 2 of the lanthanide dataset builder. Reads ``geometry_specs.csv`` from
``build_dataset_no3d.py`` and, for each *planned* spec, asks Architector to
assemble the Ln(III)-ligand complex (placing the donor atoms named in COORDLIST
around the metal, saturating the rest with the inner-sphere fill ligand:
bidentate nitrate or water) and relaxes it with GFN2-xTB. One geometry is built
per unique physical complex (build_id), never one per raw dataset row.

This builder is INCREMENTAL and CRASH-SAFE -- Architector / OpenBabel / xTB can
crash at native-code level, so a single bad complex must never kill the run:

* Subprocess isolation -- each complex is assembled in a child process, so a
  native crash (SIGSEGV/SIGABRT) kills only that complex. A per-complex timeout
  bounds runaway structures.
* Per-row append -- every attempt's status row is flushed + fsynced to the shard
  index immediately, so a shard that dies never orphans its finished geometries.
* Atomic writes -- geometry is written to a temp file and renamed only on a
  successful QC pass, so a crash mid-write never leaves a corrupt .xyz.
* Skip-existing (default) -- a valid .xyz on disk is recorded existing_ok and
  never recomputed.
* Recovery -- ``--recover-index-only`` rebuilds index rows from existing .xyz by
  matching the build_id embedded in each filename back to the specs.
* ligType / big-ligand handling -- Architector's "Cannot assign lig ... to any
  ligType!" (common for bulky floppy amide extractants) is caught, classified,
  cached, and skipped; hard ligands use deeper build profiles and explicit
  ligType overrides, never a chemically different ligand.

Run sharded (locally or on a cluster array):
    python scripts/build_unique_geometries.py --num-shards 64 --shard-id 0
    python scripts/build_unique_geometries.py --recover-index-only
    python scripts/build_unique_geometries.py --merge-index-only --num-shards 64
    python scripts/build_unique_geometries.py --audit-xyz

Regenerate only nearest-coreCN FAIL_LONG_BOND geometries, without overwriting
the original geometry directory:
    python scripts/build_unique_geometries.py regenerate-failed \
        --input reports/geometry_regenerate_fail_long_bond.csv \
        --max-attempts 3 --fixed-coordlist --rerun-qc

Verify deterministic FAIL_LONG_BOND shard assignment before submitting SLURM:
    python scripts/build_unique_geometries.py plan-regeneration-shards \
        --input reports/geometry_regenerate_fail_long_bond.csv --num-shards 64

Merge regeneration shard reports after a SLURM array run:
    python scripts/build_unique_geometries.py merge-regenerated --num-shards 64

Prepare one QC-diagnosed adjacent-CN/fill hypothesis per eligible failure:
    python scripts/build_unique_geometries.py prepare-adaptive-regeneration \
        --adaptive-input reports/family_regeneration_runs/template_pincer/regenerated_fail_long_bond_still_failed.csv

Rebuild nearest-coreCN QC queues after new geometries are merged:
    python scripts/build_unique_geometries.py triage-nearest-corecn
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import geometry_schema as gschema  # noqa: E402
from src.chemistry.coordination import (  # noqa: E402
    choose_n_ligs_for_donor_set,
    complex_build_id,
    core_cn_for_donor_set,
    detect_donors,
    n_fill as coordination_n_fill,
)

PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
AUDIT_DIR = PROCESSED_DIR / "audit"
GEOMETRY_DIR = _REPO_ROOT / "data" / "geometries"
REPORTS_DIR = _REPO_ROOT / "reports"
FAILURE_LOG_DIR = _REPO_ROOT / "logs" / "geometry_failures"
DEFAULT_SPECS = PROCESSED_DIR / "geometry_specs.csv"
MERGE_INDEX_FILE = PROCESSED_DIR / "geometry_index_for_merge.csv"
FINAL_MERGED_INDEX_FILE = PROCESSED_DIR / "geometry_index_merged.csv"
UNSUCCESSFUL_INDEX_FILE = PROCESSED_DIR / "geometry_index_unsuccessful.csv"
RECOVERED_INDEX_FILE = PROCESSED_DIR / "geometry_index_shard_recovered.csv"
KNOWN_BAD_FILE = PROCESSED_DIR / "known_bad_ligtype.csv"
LIGTYPE_OVERRIDES_FILE = PROCESSED_DIR / "ligtype_overrides.csv"
PRESCREEN_FILE = AUDIT_DIR / "ligand_prescreen.csv"
PRESCREEN_SUMMARY_FILE = AUDIT_DIR / "ligand_prescreen_summary.json"
REGENERATED_FAIL_LONG_BOND_DIR = PROCESSED_DIR / "geometries_regenerated_fail_long_bond"
DEFAULT_REGENERATE_FAIL_LONG_BOND_INPUT = REPORTS_DIR / "geometry_regenerate_fail_long_bond.csv"
DEFAULT_REGENERATE_STILL_FAILED_INPUT = REPORTS_DIR / "regenerated_fail_long_bond_still_failed.csv"
DEFAULT_MISSING_REGENERATION_QUEUE = REPORTS_DIR / "geometry_regenerate_missing_geometry_rescue.csv"
DEFAULT_HARD10_NO_STRUCTURES_QUEUE = REPORTS_DIR / "hard10_no_structures_queue.csv"
DEFAULT_HARD10_NATIVE_CRASH_QUEUE = REPORTS_DIR / "hard10_native_crash_queue.csv"
DEFAULT_FAMILY_REGENERATION_DIR = REPORTS_DIR / "family_regeneration"
DEFAULT_ADAPTIVE_REGENERATION_INPUT = DEFAULT_REGENERATE_STILL_FAILED_INPUT
DEFAULT_ADAPTIVE_REGENERATION_QUEUE = (
    DEFAULT_FAMILY_REGENERATION_DIR / "adaptive_cn_fill.csv"
)
ADAPTIVE_CN_FILL_VERSION = "adaptive_cn_fill_v1"
DEFAULT_ALL_REMAINING_PLAN_DIR = REPORTS_DIR / "all_remaining_regeneration"
DEFAULT_ALL_REMAINING_RUNS_DIR = REPORTS_DIR / "all_remaining_regeneration_runs"
DEFAULT_ALL_REMAINING_OUT_DIR = PROCESSED_DIR / "geometries_all_remaining"
ALL_REMAINING_VERSION = "all_remaining_v1"

# Job 5981236 left exactly these two independent five-row rescue groups.  Keep
# the order stable so a five-task array assigns one known molecule per task.
HARD10_NO_STRUCTURES_BUILD_IDS = (
    "f3f070427c1f",
    "bd11f8c14383",
    "033b4c93146d",
    "3927fc1f583b",
    "b63ff46d0633",
)
HARD10_NATIVE_CRASH_BUILD_IDS = (
    "3dd657f759a8",
    "c25a798202a1",
    "c6b5c3f7cc44",
    "74b654cc8157",
    "b7e4a26801b1",
)

REGENERATE_ATTEMPTS_NAME = "regenerated_fail_long_bond_attempts.csv"
REGENERATE_ACCEPTED_NAME = "regenerated_fail_long_bond_accepted.csv"
REGENERATE_STILL_FAILED_NAME = "regenerated_fail_long_bond_still_failed.csv"
REGENERATE_SUMMARY_NAME = "regenerated_fail_long_bond_summary.txt"
REGENERATE_RUN_META_NAME = "regenerated_run_meta.json"
ALL_REMAINING_PLAN_META_NAME = "plan_meta.json"

LANTHANIDE_SYMBOLS = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}
Z_BY_LANTHANIDE_SYMBOL = {
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
    "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71,
}
DONOR_QC_SYMBOLS = {"O", "N", "S", "P", "F", "Cl", "Br", "I"}
DEFAULT_LONG_BOND_THRESHOLD = 3.10
DEFAULT_BORDERLINE_LONGISH_THRESHOLD = 2.95
DEFAULT_AMBIGUOUS_GAP_THRESHOLD = 0.10

REGEN_PROVENANCE_FIELDS = [
    "run_id", "queue_sha256", "strategy_sha256", "root_source_build_id",
    "source_build_id", "parent_build_id", "rescue_route", "hypothesis_version",
]

REGEN_ATTEMPT_FIELDS = [
    "queue_index", "build_id", *REGEN_PROVENANCE_FIELDS,
    "metal_symbol", "Atomic Number_metal", "metal_ox",
    "smiles_for_architector_used", "COORDLIST", "DENTATE", "coreCN", "n_ligs",
    "inner_sphere_anion", "fill_ligand", "n_fill", "geometry_key",
    "source_xyz_path", "attempt", "profile",
    "seed", "n_symmetries", "n_conformers", "xtb_max_iterations", "returncode",
    "relax", "assemble_method", "full_method", "force_generation", "ff_preopt",
    "attempt_status", "accepted_for_clean_3d_features", "ligtype_override", "generated_xyz_path",
    "generated_mol2_path", "energy_eV", "note", "file_qc_status", "file_qc_note",
    "qc_class", "qc_note",
    "nearest_coreCN_sig", "coreCN_max_dist", "next_donor_dist",
    "gap_after_coreCN", "all_nearest",
]

REGEN_ACCEPTED_FIELDS = [
    "queue_index", gschema.BUILD_ID, *REGEN_PROVENANCE_FIELDS,
    "metal_symbol", "Atomic Number_metal", "metal_ox",
    "smiles_for_architector_used", "COORDLIST", "DENTATE", "coreCN", "n_ligs",
    "inner_sphere_anion", "fill_ligand", "n_fill", "geometry_key",
    "accepted_attempt", "accepted_profile", "accepted_ligtype_override",
    "relax", "assemble_method", "full_method", "force_generation", "ff_preopt",
    gschema.ACCEPTED_XYZ_PATH,
    gschema.ACCEPTED_MOL2_PATH, gschema.QC_CLASS, gschema.CORECN_MAX_DIST, gschema.GAP_AFTER_CORECN,
    "nearest_coreCN_sig", "file_qc_status", "file_qc_note",
    "accepted_for_clean_3d_features", "ligtype_override",
]

REGEN_STILL_FAILED_FIELDS = [
    "queue_index", "build_id", *REGEN_PROVENANCE_FIELDS,
    "metal_symbol", "Atomic Number_metal", "metal_ox",
    "smiles_for_architector_used", "COORDLIST", "DENTATE", "coreCN", "n_ligs",
    "inner_sphere_anion", "fill_ligand", "n_fill", "geometry_key",
    "attempts_run", "best_qc_class", "best_file_qc_status",
    "best_coreCN_max_dist", "best_gap_after_coreCN", "best_xyz_path",
    "best_failed_xyz_path", "failure_note", "accepted_for_clean_3d_features", "ligtype_override",
]

CHILD_SPEC_FIELDS = [
    "build_id", "reference", "Atomic Number_metal", "metal_symbol", "metal_ox",
    "canonical_smiles", "SMILES_FOR_ARCHITECTOR", "COORDLIST", "DONOR_TYPES",
    "DENTATE", "coreCN", "n_ligs", "inner_sphere_anion", "fill_ligand",
    "n_fill", "geometry_key",
]

# Inner-sphere fill ligand -> Architector's registered ligand name. nitrate must
# use the registered bidentate template name, NOT a raw nitrate SMILES.
FILL_SPECIES = {"water": "water", "nitrate": "nitrate_bi"}

# Child exit codes that classify the build outcome for the parent.
EXIT_OK = 0
EXIT_QC = 5               # parent-side sentinel: build succeeded but QC rejected it
EXIT_LIGTYPE = 7          # "Cannot assign lig ... to any ligType!"
EXIT_NO_STRUCTURE = 8     # build_complex returned nothing
EXIT_BUILD_ERROR = 9      # any other exception inside the build
EXIT_WALLTIME_BUDGET = 125  # parent stopped before SLURM could kill the shard

# Heavy-atom count above which a spec is treated as a bulky ligand and gets the
# deeper fallback profile sequence automatically.
HEAVY_LIGAND_ATOM_THRESHOLD = 60

# Merge-index schema (status/note get renamed to geometry_* by the dataset merge).
INDEX_FIELDS = [
    "build_id", "reference", "Atomic Number_metal", "metal_symbol", "metal_ox",
    "SMILES_FOR_ARCHITECTOR", "COORDLIST", "DONOR_TYPES", "DENTATE", "coreCN",
    "n_ligs", "inner_sphere_anion", "fill_ligand", "n_fill",
    "xyz_path", "mol2_path", "sdf_path", "energy_eV", "status", "note",
    "qc_status", "smiles_for_architector_original", "smiles_for_architector_used",
    "simplified_ligand", "ligtype_override",
]

# Settled, successful (or deliberately-skipped) outcomes: never recomputed.
KEEP_STATUSES = {
    "ok", "existing_ok",
    "skipped_known_bad_ligtype", "skipped_invalid_coordlist", "failed_ligtype",
}
# Statuses re-attempted only when explicitly requested via --retry-* flags.
RETRYABLE_STATUSES = {
    "failed_timeout", "failed_native_crash", "failed_native_crash_repeated",
    "failed_no_structures", "failed_ligtype", "failed_exception",
    "failed_invalid_xyz", "failed_qc", "failed_walltime_budget",
    "skipped_known_bad_ligtype",
    "ok_simplified_ligand",
}
SUCCESS_STATUSES = ("ok", "existing_ok")
# These statuses may have a syntactically valid XYZ at the canonical path, but
# that artifact is precisely what an explicit retry is asking us to replace.
FORCE_REBUILD_STATUSES = {"ok_simplified_ligand", "failed_qc"}

# Cheap geometry-QC failure labels (compatible with the broader pipeline set).
QC_LABELS = {
    "accepted", "QC_FAILED", "BAD_GEOMETRY", "METAL_NOT_COORDINATED",
    "NITRATE_MISSING", "WATER_MISSING", "COMPOSITION_MISMATCH", "FRAGMENTED",
}

BUILD_PROFILES = {
    "standard": {
        "n_symmetries": None, "n_conformers": None, "ff_preopt": True,
        "relax": True, "xtb_max_iterations": 250, "seed_offset": 0,
    },
    # Bulky / floppy ligands often fail on an unlucky placement lottery. Broaden
    # the conformer search and xTB budget.
    "large_ligand_deep": {
        "n_symmetries": 80, "n_conformers": 12, "ff_preopt": True,
        "relax": True, "xtb_max_iterations": 500, "seed_offset": 1009,
    },
    # A smaller search rescues cases that time out / enter a bad native path in
    # the deep profile.
    "large_ligand_fast": {
        "n_symmetries": 20, "n_conformers": 3, "ff_preopt": True,
        "relax": True, "xtb_max_iterations": 350, "seed_offset": 2027,
    },
    # Last-resort Architector path for no_structures/native-xTB failures.  It
    # still runs Architector's geometry sanity checks, but avoids an expensive
    # final xTB relaxation and permits UFF evaluation when xTB cannot provide
    # energies.  This is deliberately a fallback, never the default profile.
    "emergency_unrelaxed": {
        "n_symmetries": 40, "n_conformers": 4, "ff_preopt": True,
        "relax": False, "xtb_max_iterations": 100, "seed_offset": 4099,
        "assemble_method": "GFN-FF", "full_method": "UFF",
        "force_generation": True,
    },
    # Targeted rescue for the large complexes that crash inside GFN-FF force
    # field initialisation.  UFF assembly plus disabled ff_preopt keeps that
    # native GFN-FF path out of the process while preserving the full ligand.
    "uff_xtb_no_preopt": {
        "n_symmetries": 40, "n_conformers": 4, "ff_preopt": False,
        "relax": True, "xtb_max_iterations": 500, "seed_offset": 8191,
        "assemble_method": "UFF", "full_method": "GFN2-xTB",
        "force_generation": True,
    },
    # If the final xTB relaxation is also unstable, retain Architector's UFF
    # geometry as a second rescue attempt and let the normal geometry QC decide.
    "uff_unrelaxed": {
        "n_symmetries": 40, "n_conformers": 4, "ff_preopt": False,
        "relax": False, "xtb_max_iterations": 100, "seed_offset": 12289,
        "assemble_method": "UFF", "full_method": "UFF",
        "force_generation": True,
    },
}
DEFAULT_PROFILE_SEQUENCE = "standard"
HARD_LIGAND_PROFILE_SEQUENCE = "standard,large_ligand_deep,large_ligand_fast"
LIGTYPE_DENTICITY = {
    "mono": 1,
    "bi_cis": 2,
    "bi_cis_bulky": 2,
    "bi_cis_chelating": 2,
    "bi_cis_planar": 2,
    "bi_trans": 2,
    "tri_fac": 3,
    "tri_mer": 3,
    "tri_mer_bent": 3,
    "tetra_planar": 4,
    "tetra_planar_bent": 4,
    "tetra_pyramidal": 4,
    "tetra_seesaw": 4,
    "tetra_trigonal_pyramidal": 4,
    "penta_planar": 5,
    "penta_planar_bent": 5,
    "penta_pyramidal": 5,
    "penta_square_pyramidal": 5,
    "hexa_octahedral": 6,
    "hexa_planar": 6,
    "hexa_trigonal_prismatic": 6,
    "hepta_5_2": 7,
    "hepta_capped_trigonal_prismatic": 7,
    "hepta_pentagonal_bipyramidal": 7,
    "octa_cubic": 8,
    "octa_square_antiprismatic": 8,
    "octa_trigonal_prismatic_triangle_face_bicapped": 8,
    "nona_capped_square_antiprismatic": 9,
}

AUTO_LIGTYPE_TOKEN = "auto"
LIGTYPE_ALTERNATIVES = {
    1: ("mono",),
    2: ("bi_cis_chelating", "bi_cis_planar", "bi_cis_bulky", "bi_cis", "bi_trans"),
    3: ("tri_mer_bent", "tri_fac", "tri_mer"),
    4: (
        "tetra_planar_bent", "tetra_trigonal_pyramidal", "tetra_seesaw",
        "tetra_pyramidal", "tetra_planar",
    ),
    5: ("penta_planar_bent", "penta_square_pyramidal", "penta_pyramidal", "penta_planar"),
    6: ("hexa_octahedral", "hexa_trigonal_prismatic", "hexa_planar"),
    7: ("hepta_capped_trigonal_prismatic", "hepta_pentagonal_bipyramidal", "hepta_5_2"),
    8: ("octa_square_antiprismatic", "octa_cubic", "octa_trigonal_prismatic_triangle_face_bicapped"),
    9: ("nona_capped_square_antiprismatic",),
}

SUPPORTED_REGENERATION_ROUTES = {
    "adaptive_cn_fill", "aminopoly_cn8", "placement_qc", "placement_build",
    "canonical_template_replan", "canonical_missing", "nitrate_parity_cn",
}


# ---------------------------------------------------------------------------
# Path / spec helpers
# ---------------------------------------------------------------------------
def _coordlist(cell) -> list[int]:
    try:
        return [int(x) for x in json.loads(cell)]
    except Exception:
        return []


def expected_paths(spec: pd.Series, out_root: Path) -> tuple[Path, Path, Path]:
    """Deterministic output paths for one spec (build_id makes them unique)."""
    Z = int(spec["Atomic Number_metal"])
    symbol = str(spec["metal_symbol"])
    metal_ox = int(spec.get("metal_ox", 3))
    core_cn = int(spec["coreCN"])
    n_ligs = int(spec["n_ligs"])
    anion = str(spec.get("inner_sphere_anion", "water"))
    build_id = str(spec["build_id"])
    base = f"{symbol}_Z{Z}_ox{metal_ox}_CN{core_cn}_nlig{n_ligs}_{anion}_{build_id}"
    out_dir = out_root / symbol
    return out_dir / f"{base}.xyz", out_dir / f"{base}.mol2", out_dir / f"{base}.sdf"


def valid_xyz(path: Path) -> bool:
    """Header count parses and every declared atom has numeric coordinates."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if len(lines) < 3:
            return False
        n = int(lines[0].strip())
        if n < 2 or len(lines) < n + 2:
            return False
        for line in lines[2:n + 2]:
            fields = line.split()
            if len(fields) < 4 or not fields[0]:
                return False
            coordinates = [float(value) for value in fields[1:4]]
            if not all(math.isfinite(value) for value in coordinates):
                return False
        return True
    except Exception:
        return False


def _expected_heavy_atom_counts(spec: pd.Series | dict) -> Counter | None:
    """Return the exact non-H composition implied by one frozen build spec.

    Architector fills nitrate sites with bidentate nitrate and uses water for an
    odd residual site.  The separate nitrate-presence gate below intentionally
    rejects an odd nitrate-only site filled solely by that secondary water.
    """
    smiles = _safe_text(spec.get("SMILES_FOR_ARCHITECTOR", ""))
    metal = _safe_text(spec.get("metal_symbol", ""))
    n_ligs = _safe_int(spec.get("n_ligs"), 0)
    n_fill = _safe_int(spec.get("n_fill"), -1)
    if not smiles or not metal or n_ligs < 1 or n_fill < 0:
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return None

    ligand = Counter(
        atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"
    )
    expected = Counter({symbol: count * n_ligs for symbol, count in ligand.items()})
    expected[metal] += 1
    fill = _safe_text(
        spec.get("fill_ligand", spec.get("inner_sphere_anion", "water")),
        "water",
    ).lower()
    if fill in {"nitrate", "nitrate_bi"}:
        nitrate_count, secondary_water_count = divmod(n_fill, 2)
        expected["N"] += nitrate_count
        expected["O"] += 3 * nitrate_count + secondary_water_count
    else:
        expected["O"] += n_fill
    return expected


def _inner_sphere_nitrate_count(atoms: list[dict], metal_symbol: str) -> int:
    """Count nitrate-like N(O)3 groups with at least one O near the metal."""
    metals = [atom for atom in atoms if atom["symbol"] == metal_symbol]
    if not metals:
        return 0
    metal = metals[0]
    count = 0
    for nitrogen in (atom for atom in atoms if atom["symbol"] == "N"):
        oxygens = [
            atom for atom in atoms
            if atom["symbol"] == "O" and _distance(nitrogen, atom) <= 1.75
        ]
        if len(oxygens) >= 3 and any(_distance(metal, oxygen) <= 3.40 for oxygen in oxygens):
            count += 1
    return count


def qc_xyz(path: Path, spec: pd.Series) -> tuple[str, str]:
    """Cheap, file-level geometry QC. Returns (qc_status, note).

    Implements the file-level checks now (file parses, metal present, enough
    atoms, fill species present when expected) and leaves TODO hooks for deeper
    geometry checks (donor-metal distances, atom clashes, fragmentation).
    """
    if not valid_xyz(path):
        return "QC_FAILED", "invalid_or_unreadable_xyz"
    atoms = _read_xyz_atoms(path)
    symbols = [atom["symbol"] for atom in atoms]
    metal = str(spec["metal_symbol"])
    if metal not in symbols:
        return "METAL_NOT_COORDINATED", f"metal_{metal}_absent_from_xyz"
    if len(symbols) < 2:
        return "FRAGMENTED", "too_few_atoms"
    anion = _safe_text(spec.get("inner_sphere_anion", "water"), "water").lower()
    n_fill = _safe_int(spec.get("n_fill"), 0)
    if anion == "nitrate" and n_fill > 0:
        required_nitrates = max(1, n_fill // 2)
        observed_nitrates = _inner_sphere_nitrate_count(atoms, metal)
        if observed_nitrates < required_nitrates:
            return (
                "NITRATE_MISSING",
                f"inner_sphere_nitrate_count={observed_nitrates};expected={required_nitrates}",
            )

    expected = _expected_heavy_atom_counts(spec)
    if expected is None:
        return "QC_FAILED", "expected_heavy_atom_composition_unavailable"
    observed = Counter(symbol for symbol in symbols if symbol != "H")
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        fill = _safe_text(
            spec.get("fill_ligand", spec.get("inner_sphere_anion", "water")),
            "water",
        ).lower()
        if fill == "water" and missing.get("O", 0) > 0:
            return "WATER_MISSING", f"missing_heavy_atoms:{dict(missing)}"
        return (
            "COMPOSITION_MISMATCH",
            f"heavy_atom_formula_mismatch:missing={dict(missing)};extra={dict(extra)}",
        )
    # TODO: connected-graph fragmentation checks once a fast bond perception
    # layer is available. Exact heavy composition and nearest-core CN are
    # intentionally separate gates.
    return "accepted", "file_level_qc_ok"


def _safe_float(value, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value, default: str = "") -> str:
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value)
    return default if text.lower() == "nan" else text


def _read_xyz_atoms(path: Path) -> list[dict]:
    """Parse XYZ symbols and coordinates from an already-written file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("xyz_has_too_few_lines")
    n_atoms = int(lines[0].strip())
    atoms: list[dict] = []
    for idx, line in enumerate(lines[2:2 + n_atoms]):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"xyz_coordinate_line_{idx}_has_too_few_fields")
        atoms.append({
            "index": idx,
            "symbol": parts[0],
            "x": float(parts[1]),
            "y": float(parts[2]),
            "z": float(parts[3]),
        })
    if len(atoms) != n_atoms:
        raise ValueError("xyz_atom_count_mismatch")
    return atoms


def _distance(a: dict, b: dict) -> float:
    return math.sqrt(
        (float(a["x"]) - float(b["x"])) ** 2
        + (float(a["y"]) - float(b["y"])) ** 2
        + (float(a["z"]) - float(b["z"])) ** 2
    )


def _fmt_dist(value: float) -> str:
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}"


def _nearest_sig(entries: list[tuple[dict, float]]) -> str:
    return ";".join(
        f"{atom['symbol']}{int(atom['index'])}:{dist:.3f}"
        for atom, dist in entries
    )


def nearest_corecn_qc_xyz(
    path: Path,
    spec: pd.Series | dict,
    *,
    long_bond_threshold: float = DEFAULT_LONG_BOND_THRESHOLD,
    borderline_longish_threshold: float = DEFAULT_BORDERLINE_LONGISH_THRESHOLD,
    ambiguous_gap_threshold: float = DEFAULT_AMBIGUOUS_GAP_THRESHOLD,
) -> dict:
    """Nearest-coreCN metal-donor QC used for targeted regeneration reports."""
    result = {
        "qc_class": "QC_FAILED",
        "qc_note": "",
        "nearest_coreCN_sig": "",
        "coreCN_max_dist": "",
        "next_donor_dist": "",
        "gap_after_coreCN": "",
        "all_nearest": "",
    }
    try:
        if not valid_xyz(path):
            result["qc_note"] = "invalid_or_unreadable_xyz"
            return result
        atoms = _read_xyz_atoms(path)
    except Exception as exc:
        result["qc_note"] = f"xyz_parse_error:{type(exc).__name__}:{str(exc)[:180]}"
        return result

    wanted_metal = _safe_text(spec.get("metal_symbol", "") if hasattr(spec, "get") else "")
    metals = [atom for atom in atoms if atom["symbol"] in LANTHANIDE_SYMBOLS]
    if wanted_metal:
        exact = [atom for atom in metals if atom["symbol"] == wanted_metal]
        if exact:
            metals = exact
    if not metals:
        result["qc_class"] = "FAIL_NO_METAL"
        result["qc_note"] = "lanthanide_metal_absent_from_xyz"
        return result
    metal = metals[0]

    core_cn = _safe_int(spec.get("coreCN", 0) if hasattr(spec, "get") else 0)
    if core_cn <= 0:
        result["qc_class"] = "FAIL_INVALID_CORECN"
        result["qc_note"] = "coreCN_missing_or_nonpositive"
        return result

    donors = [
        (atom, _distance(metal, atom))
        for atom in atoms
        if atom["symbol"] in DONOR_QC_SYMBOLS
    ]
    donors.sort(key=lambda item: item[1])
    result["all_nearest"] = _nearest_sig(donors)

    if len(donors) < core_cn:
        nearest = donors[:core_cn]
        result["qc_class"] = "FAIL_TOO_FEW_DONORS"
        result["nearest_coreCN_sig"] = _nearest_sig(nearest)
        if nearest:
            result["coreCN_max_dist"] = _fmt_dist(max(dist for _, dist in nearest))
        result["qc_note"] = f"only_{len(donors)}_donor_atoms_for_coreCN_{core_cn}"
        return result

    nearest = donors[:core_cn]
    core_cn_max = max(dist for _, dist in nearest)
    next_donor_dist = donors[core_cn][1] if len(donors) > core_cn else math.inf
    gap = next_donor_dist - core_cn_max

    if core_cn_max > float(long_bond_threshold):
        qc_class = "FAIL_LONG_BOND"
        note = f"coreCN_max_dist>{float(long_bond_threshold):.2f}"
    elif gap < float(ambiguous_gap_threshold):
        qc_class = "BORDERLINE_AMBIGUOUS_SHELL"
        note = f"gap_after_coreCN<{float(ambiguous_gap_threshold):.2f}"
    elif core_cn_max > float(borderline_longish_threshold):
        qc_class = "BORDERLINE_LONGISH"
        note = f"coreCN_max_dist>{float(borderline_longish_threshold):.2f}"
    else:
        qc_class = "OK"
        note = "nearest_coreCN_qc_ok"

    result.update({
        "qc_class": qc_class,
        "qc_note": note,
        "nearest_coreCN_sig": _nearest_sig(nearest),
        "coreCN_max_dist": _fmt_dist(core_cn_max),
        "next_donor_dist": _fmt_dist(next_donor_dist),
        "gap_after_coreCN": _fmt_dist(gap),
    })
    return result


def index_row(spec: pd.Series, *, status: str, note: str = "", qc_status: str = "",
              xyz_path: str = "", mol2_path: str = "", sdf_path: str = "",
              energy_eV: str = "", smiles_used: str | None = None,
              simplified_ligand: bool = False, ligtype_override: str = "") -> dict:
    original_smiles = str(spec["SMILES_FOR_ARCHITECTOR"])
    return {
        "build_id": str(spec["build_id"]),
        "reference": spec.get("reference", ""),
        "Atomic Number_metal": int(spec["Atomic Number_metal"]),
        "metal_symbol": str(spec["metal_symbol"]),
        "metal_ox": int(spec.get("metal_ox", 3)),
        "SMILES_FOR_ARCHITECTOR": original_smiles,
        "COORDLIST": spec["COORDLIST"],
        "DONOR_TYPES": spec.get("DONOR_TYPES", ""),
        "DENTATE": spec.get("DENTATE", ""),
        "coreCN": int(spec["coreCN"]),
        "n_ligs": int(spec["n_ligs"]),
        "inner_sphere_anion": str(spec.get("inner_sphere_anion", "water")),
        "fill_ligand": str(spec.get("fill_ligand", spec.get("inner_sphere_anion", "water"))),
        "n_fill": spec.get("n_fill", ""),
        "xyz_path": xyz_path,
        "mol2_path": mol2_path,
        "sdf_path": sdf_path,
        "energy_eV": energy_eV,
        "status": status,
        "note": note,
        "qc_status": qc_status,
        "smiles_for_architector_original": original_smiles,
        "smiles_for_architector_used": smiles_used or original_smiles,
        "simplified_ligand": bool(simplified_ligand),
        "ligtype_override": ligtype_override,
    }


def append_index_row(csv_path: Path, row: dict) -> None:
    """Append one status row to a shard index, header once, flushed + fsynced."""
    if csv_path.exists():
        upgrade_index_csv_schema(csv_path)
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _coerce_index_rows(csv_path: Path) -> list[dict]:
    """Read legacy/mixed index CSV rows and normalize them to INDEX_FIELDS."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []

        rows = []
        header_has_ligtype = "ligtype_override" in header
        for raw in reader:
            row = dict(zip(header, raw))
            if len(raw) > len(header):
                extras = raw[len(header):]
                if not header_has_ligtype and len(extras) == 1:
                    row["ligtype_override"] = extras[0]
                else:
                    row["_extra_fields"] = json.dumps(extras)
            for field in INDEX_FIELDS:
                row.setdefault(field, "")
            rows.append({field: row.get(field, "") for field in INDEX_FIELDS})
    return rows


def upgrade_index_csv_schema(csv_path: Path) -> None:
    """Upgrade old shard-index headers after INDEX_FIELDS grows."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return
    if header == INDEX_FIELDS:
        return

    rows = _coerce_index_rows(csv_path)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, csv_path)


def read_index_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, low_memory=False).fillna("")
    except pd.errors.ParserError:
        return pd.DataFrame(_coerce_index_rows(csv_path), columns=INDEX_FIELDS)



def _validated_index_tag(index_tag: str = "") -> str:
    tag = str(index_tag or "").strip()
    if tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise SystemExit(
            "--index-tag may contain only letters, digits, dot, underscore, and hyphen"
        )
    return tag


def _shard_artifact_name(prefix: str, shard_id: int, num_shards: int,
                         index_tag: str = "") -> str:
    tag = _validated_index_tag(index_tag)
    tag_part = f"_{tag}" if tag else ""
    return f"{prefix}{tag_part}_shard{shard_id}of{num_shards}.csv"


def shard_index_parts(num_shards: int = 16, index_tag: str = "") -> list[Path]:
    tag = _validated_index_tag(index_tag)
    if tag:
        pattern = rf"geometry_index_{re.escape(tag)}_shard\d+of{num_shards}\.csv"
        return sorted(
            p for p in PROCESSED_DIR.glob(f"geometry_index_{tag}_shard*of{num_shards}.csv")
            if re.fullmatch(pattern, p.name)
        )
    exact = [
        p for p in PROCESSED_DIR.glob(f"geometry_index_shard*of{num_shards}.csv")
        if re.fullmatch(rf"geometry_index_shard\d+of{num_shards}\.csv", p.name)
    ]
    if exact:
        return sorted(exact)
    return sorted(
        p for p in PROCESSED_DIR.glob("geometry_index_shard*of*.csv")
        if re.fullmatch(r"geometry_index_shard\d+of\d+\.csv", p.name)
    )


def current_merged_index_file() -> Path:
    return FINAL_MERGED_INDEX_FILE if FINAL_MERGED_INDEX_FILE.exists() else MERGE_INDEX_FILE


# ---------------------------------------------------------------------------
# coordList validation + ligand pre-screen
# ---------------------------------------------------------------------------
def validate_coordlist(smiles: str, coord: list[int], dentate) -> tuple[bool, str]:
    try:
        from rdkit import Chem
    except Exception:
        return True, "rdkit_unavailable_skipped_check"
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False, "SMILES_PARSE_ERROR"
    if not coord:
        return False, "NO_DONOR_ATOMS"
    n = mol.GetNumAtoms()
    if any((i < 0 or i >= n) for i in coord):
        return False, "INVALID_DONOR_INDICES"
    if len(set(coord)) != len(coord):
        return False, "INVALID_DONOR_INDICES"
    syms = {mol.GetAtomWithIdx(i).GetSymbol() for i in coord}
    if not syms <= {"O", "N", "S", "P"}:
        return False, f"non_donor_atom:{sorted(syms)}"
    try:
        if dentate not in (None, "") and len(coord) != int(float(dentate)):
            return False, "DENTICITY_MISMATCH"
    except (TypeError, ValueError):
        pass
    return True, "ok"


def ligand_metrics(smiles: str) -> dict:
    """Heavy-atom count, MW, rotatable bonds for the big-ligand pre-screen."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except Exception:
        return {}
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {}
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mol_weight": round(float(Descriptors.MolWt(mol)), 2),
        "n_rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
    }


# ---------------------------------------------------------------------------
# Legacy ligand simplification helpers, retained only for old-row provenance.
# Production generation must use the original ligand from SMILES_FOR_ARCHITECTOR.
# ---------------------------------------------------------------------------
_LONG_ALKYL_RE = re.compile(r"C{4,}")


def _canonical_smiles_if_possible(smiles: str) -> str | None:
    try:
        from rdkit import Chem
    except Exception:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None


def simplify_long_alkyl_tails(smiles: str) -> str | None:
    """Shorten long aliphatic tails while keeping donor-rich functional groups.

    Targets repeated uppercase aliphatic carbon runs only; aromatic atoms,
    carbonyls, heteroatoms, charges and ring notation are left untouched.
    """
    simplified = _LONG_ALKYL_RE.sub("C", str(smiles))
    if simplified == str(smiles):
        simplified = re.sub(r"C{3,}", "C", str(smiles))
    return _canonical_smiles_if_possible(simplified)


def _donor_symbols(smiles: str, coord: list[int]) -> list[str] | None:
    try:
        from rdkit import Chem
    except Exception:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    syms = []
    for idx in coord:
        if idx < 0 or idx >= mol.GetNumAtoms():
            return None
        syms.append(mol.GetAtomWithIdx(idx).GetSymbol())
    return syms


def _coordlist_for_simplified(original_smiles, original_coord, simplified_smiles) -> list[int] | None:
    donor_symbols = _donor_symbols(original_smiles, original_coord)
    if not donor_symbols:
        return None
    try:
        from rdkit import Chem
    except Exception:
        return None
    mol = Chem.MolFromSmiles(str(simplified_smiles))
    if mol is None:
        return None
    donor_atoms = [(a.GetIdx(), a.GetSymbol()) for a in mol.GetAtoms()
                   if a.GetSymbol() in {"O", "N", "S", "P"}]
    used: set[int] = set()
    coord: list[int] = []
    for symbol in donor_symbols:
        match = next((idx for idx, sym in donor_atoms if sym == symbol and idx not in used), None)
        if match is None:
            return None
        used.add(match)
        coord.append(match)
    return coord


def simplified_ligand_spec(spec: pd.Series) -> tuple[pd.Series | None, str]:
    original_smiles = str(spec["SMILES_FOR_ARCHITECTOR"])
    original_coord = _coordlist(spec["COORDLIST"])
    simplified_smiles = simplify_long_alkyl_tails(original_smiles)
    if not simplified_smiles:
        return None, "simplified_smiles_unparseable"
    if simplified_smiles == original_smiles:
        return None, "simplified_smiles_unchanged"
    simplified_coord = _coordlist_for_simplified(original_smiles, original_coord, simplified_smiles)
    if not simplified_coord:
        return None, "simplified_coordlist_unavailable"
    ok, note = validate_coordlist(simplified_smiles, simplified_coord, spec.get("DENTATE"))
    if not ok:
        return None, f"simplified_coordlist_invalid:{note}"
    out = spec.copy()
    out["SMILES_FOR_ARCHITECTOR"] = simplified_smiles
    out["COORDLIST"] = json.dumps(simplified_coord)
    return out, "ok"


# ---------------------------------------------------------------------------
# known-bad ligType cache (shared across shards, best-effort)
# ---------------------------------------------------------------------------
def _bad_key(smiles: str, coordlist_cell) -> str:
    return f"{str(smiles).strip()}|{str(coordlist_cell).strip()}"


def _coordlist_key(coordlist_cell) -> str:
    coord = _coordlist(coordlist_cell)
    return json.dumps(coord, separators=(",", ":")) if coord else str(coordlist_cell).strip()


def _smiles_coord_key(smiles: str, coordlist_cell) -> str:
    return f"{str(smiles).strip()}|{_coordlist_key(coordlist_cell)}"


def load_known_bad() -> set[str]:
    if not KNOWN_BAD_FILE.exists():
        return set()
    try:
        df = pd.read_csv(KNOWN_BAD_FILE)
        return {_bad_key(r["SMILES_FOR_ARCHITECTOR"], r["COORDLIST"]) for _, r in df.iterrows()}
    except Exception:
        return set()


def append_known_bad(smiles: str, coordlist_cell) -> None:
    new_file = not KNOWN_BAD_FILE.exists()
    try:
        with open(KNOWN_BAD_FILE, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["SMILES_FOR_ARCHITECTOR", "COORDLIST"])
            if new_file:
                writer.writeheader()
            writer.writerow({"SMILES_FOR_ARCHITECTOR": smiles, "COORDLIST": coordlist_cell})
    except Exception:
        pass


def _first_column(row: pd.Series, names: list[str]) -> str:
    for name in names:
        if name in row.index:
            value = _safe_text(row.get(name, ""))
            if value:
                return value
    return ""


def load_ligtype_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    overrides = {"by_build_id": {}, "by_geometry_key": {}, "by_smiles_coord": {}}
    if path is None or not Path(path).exists():
        return overrides
    table = pd.read_csv(path, low_memory=False)
    for row_num, row in table.iterrows():
        enabled = _first_column(row, ["enabled", "active", "use"])
        if enabled and enabled.strip().lower() in {"0", "false", "no", "n"}:
            continue
        ligtype = _first_column(row, ["ligType", "ligtype", "lig_type"])
        if not ligtype:
            raise SystemExit(f"{path}: row {row_num + 2} missing ligType")
        if ligtype not in LIGTYPE_DENTICITY:
            choices = ", ".join(sorted(LIGTYPE_DENTICITY))
            raise SystemExit(f"{path}: row {row_num + 2} unknown ligType {ligtype!r}. Choose from: {choices}")
        build_id = _first_column(row, ["build_id", "geometry_build_id"])
        geometry_key = _first_column(row, ["geometry_key"])
        smiles = _first_column(row, ["smiles_for_architector_used", "SMILES_FOR_ARCHITECTOR", "smiles"])
        coordlist = _first_column(row, ["COORDLIST", "coordList", "coordlist"])
        record = {
            "ligType": ligtype,
            "source_row": str(row_num + 2),
            "note": _first_column(row, ["note", "reason", "review_note"]),
        }
        registered = False
        if build_id:
            overrides["by_build_id"][str(build_id)] = record
            registered = True
        if geometry_key:
            overrides["by_geometry_key"][str(geometry_key)] = record
            registered = True
        if smiles and coordlist:
            overrides["by_smiles_coord"][_smiles_coord_key(smiles, coordlist)] = record
            registered = True
        if not registered:
            raise SystemExit(
                f"{path}: row {row_num + 2} needs build_id, geometry_key, "
                "or smiles_for_architector_used + COORDLIST"
            )
    return overrides


def _ligtype_override_for_spec(spec: pd.Series, args) -> str:
    overrides = getattr(args, "ligtype_override_index", None) or {}
    if not overrides:
        return ""
    record = None
    build_id = _safe_text(spec.get("build_id", ""))
    geometry_key = _safe_text(spec.get("geometry_key", ""))
    smiles = _safe_text(spec.get("SMILES_FOR_ARCHITECTOR", ""))
    coordlist = _safe_text(spec.get("COORDLIST", ""))
    if build_id:
        record = overrides.get("by_build_id", {}).get(build_id)
    if record is None and geometry_key:
        record = overrides.get("by_geometry_key", {}).get(geometry_key)
    if record is None and smiles and coordlist:
        record = overrides.get("by_smiles_coord", {}).get(_smiles_coord_key(smiles, coordlist))
    if record is None:
        return ""
    ligtype = record["ligType"]
    expected = _safe_int(spec.get("DENTATE", 0), default=0) or len(_coordlist(coordlist))
    actual = LIGTYPE_DENTICITY.get(ligtype)
    if expected and actual and expected != actual:
        raise SystemExit(
            f"ligType override mismatch for build_id={build_id}: "
            f"{ligtype} denticity={actual}, expected DENTATE/COORDLIST={expected}"
        )
    return ligtype


def _ligtype_candidates_for_spec(spec: pd.Series, args) -> list[str]:
    sequence = _safe_text(spec.get("ligtype_sequence", ""))
    if not sequence:
        return [_ligtype_override_for_spec(spec, args)]
    candidates = [token.strip() for token in re.split(r"[:,]", sequence) if token.strip()]
    if not candidates:
        return [_ligtype_override_for_spec(spec, args)]
    expected = _safe_int(spec.get("DENTATE"), 0) or len(_coordlist(spec.get("COORDLIST", "")))
    for candidate in candidates:
        if candidate == AUTO_LIGTYPE_TOKEN:
            continue
        if candidate not in LIGTYPE_DENTICITY:
            raise SystemExit(f"Unknown ligType candidate {candidate!r}")
        if expected and LIGTYPE_DENTICITY[candidate] != expected:
            raise SystemExit(
                f"ligType candidate mismatch for build_id={spec.get('build_id')}: "
                f"{candidate} denticity={LIGTYPE_DENTICITY[candidate]}, expected={expected}"
            )
    return candidates


def _alternative_ligtype_sequence(
    spec: pd.Series | dict,
    current_ligtype: str = "",
    *,
    include_current: bool = False,
    max_candidates: int = 3,
) -> str:
    denticity = _safe_int(spec.get("DENTATE"), 0)
    alternatives = list(LIGTYPE_ALTERNATIVES.get(denticity, ()))
    ordered: list[str] = []
    if include_current and current_ligtype:
        ordered.append(current_ligtype)
    ordered.extend(value for value in alternatives if value != current_ligtype)
    ordered.append(AUTO_LIGTYPE_TOKEN)
    deduped = list(dict.fromkeys(ordered))[:max(1, int(max_candidates))]
    return ":".join(deduped)


def _profile_names(sequence: str) -> list[str]:
    names = [n.strip() for n in str(sequence).split(",") if n.strip()]
    unknown = [n for n in names if n not in BUILD_PROFILES]
    if unknown:
        raise SystemExit(f"Unknown build profile(s): {unknown}. Choose from {sorted(BUILD_PROFILES)}")
    return names or ["standard"]


def _profile_value(profile: dict, key: str, default):
    value = profile.get(key)
    return default if value is None else value


# ---------------------------------------------------------------------------
# CHILD: build exactly one complex in an isolated process
# ---------------------------------------------------------------------------
def _build_one_child(args) -> int:
    if getattr(args, "spec_json", None):
        try:
            spec = pd.Series(json.loads(args.spec_json))
        except Exception as exc:
            print(f"__RESULT__ {json.dumps({'note': f'spec_json_parse_failed: {exc}'})}")
            return EXIT_BUILD_ERROR
    else:
        specs = pd.read_csv(args.specs, low_memory=False)
        match = specs[specs["build_id"].astype(str) == str(args.build_id)]
        if match.empty:
            print(f"__RESULT__ {json.dumps({'note': 'build_id_not_in_specs'})}")
            return EXIT_BUILD_ERROR
        spec = match.iloc[0]

    smiles = str(args.override_smiles or spec["SMILES_FOR_ARCHITECTOR"])
    coord = (_coordlist(args.override_coordlist)
             if args.override_coordlist else _coordlist(spec["COORDLIST"]))
    ligtype_override = _safe_text(getattr(args, "override_ligtype", ""))
    symbol = str(spec["metal_symbol"])
    core_cn = int(spec["coreCN"])
    n_ligs = int(spec["n_ligs"])
    metal_ox = int(spec.get("metal_ox", 3))
    anion = str(spec.get("inner_sphere_anion", "water"))
    fill_species = FILL_SPECIES.get(anion, "water")

    xyz_path, mol2_path, _ = expected_paths(spec, Path(args.out))
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_xyz = xyz_path.with_suffix(".tmp.xyz")
    tmp_xyz.unlink(missing_ok=True)

    try:
        from architector import build_complex
    except Exception as exc:
        print(f"__RESULT__ {json.dumps({'note': f'architector_import_failed: {exc}'})}")
        return EXIT_BUILD_ERROR

    profile = BUILD_PROFILES[args.profile]
    n_symmetries = int(_profile_value(profile, "n_symmetries", args.n_symmetries))
    n_conformers = int(_profile_value(profile, "n_conformers", args.n_conformers))
    xtb_max_iterations = int(_profile_value(profile, "xtb_max_iterations", args.xtb_max_iterations))
    seed = int(args.seed) + int(profile.get("seed_offset", 0))

    ligand_dict = {"smiles": smiles, "coordList": coord}
    if ligtype_override:
        ligand_dict["ligType"] = ligtype_override

    input_dict = {
        "core": {"metal": symbol, "coreCN": core_cn},
        "ligands": [dict(ligand_dict) for _ in range(n_ligs)],
        "parameters": {
            "metal_ox": metal_ox,
            "n_symmetries": n_symmetries,
            "n_conformers": n_conformers,
            "return_only_1": True,
            "relax": bool(profile.get("relax", True)),
            "assemble_method": str(profile.get("assemble_method", "GFN2-xTB")),
            "full_method": str(profile.get("full_method", "GFN2-xTB")),
            "force_generation": bool(profile.get("force_generation", False)),
            "ff_preopt": bool(profile.get("ff_preopt", True)),
            "xtb_solvent": "none",
            "xtb_max_iterations": xtb_max_iterations,
            "fill_ligand": fill_species,
            "secondary_fill_ligand": "water",
            "seed": seed,
        },
    }

    try:
        out = build_complex(input_dict)
    except ValueError as exc:
        if "Cannot assign lig" in str(exc):
            print(f"__RESULT__ {json.dumps({'note': str(exc)[:300]})}")
            return EXIT_LIGTYPE
        traceback.print_exc()
        print(f"__RESULT__ {json.dumps({'note': f'ValueError: {str(exc)[:300]}'})}")
        return EXIT_BUILD_ERROR
    except Exception as exc:
        traceback.print_exc()
        print(f"__RESULT__ {json.dumps({'note': f'{type(exc).__name__}: {str(exc)[:300]}'})}")
        return EXIT_BUILD_ERROR

    if not out:
        print(f"__RESULT__ {json.dumps({'note': 'no_structures_returned'})}")
        return EXIT_NO_STRUCTURE

    best = out[next(iter(out))]
    try:
        best["ase_atoms"].write(str(tmp_xyz))            # write to temp first
        if not valid_xyz(tmp_xyz):
            raise ValueError("written_xyz_failed_validation")
        os.replace(tmp_xyz, xyz_path)                    # atomic promotion
        try:
            if best.get("mol2string"):
                mol2_path.write_text(best["mol2string"])
        except Exception:
            pass
        energy = best.get("energy", "")
        print(f"__RESULT__ {json.dumps({'energy_eV': str(energy), 'profile': args.profile, 'ligtype_override': ligtype_override})}")
        return EXIT_OK
    except Exception as exc:
        tmp_xyz.unlink(missing_ok=True)
        traceback.print_exc()
        print(f"__RESULT__ {json.dumps({'note': f'write_failed: {type(exc).__name__}: {exc}'})}")
        return EXIT_BUILD_ERROR


# ---------------------------------------------------------------------------
# PARENT: dispatch one spec to an isolated child, classify the outcome
# ---------------------------------------------------------------------------
def _result_note(stdout: str, stderr: str) -> tuple[str, str]:
    note, energy = "", ""
    for line in stdout.splitlines():
        if line.startswith("__RESULT__"):
            try:
                payload = json.loads(line[len("__RESULT__"):].strip())
                note = payload.get("note", "")
                energy = payload.get("energy_eV", "")
            except Exception:
                pass
    if not note and stderr.strip():
        note = stderr.strip().splitlines()[-1][:300]
    return note, energy


def _heavy_ligand(spec: pd.Series) -> bool:
    metrics = ligand_metrics(str(spec["SMILES_FOR_ARCHITECTOR"]))
    return int(metrics.get("heavy_atoms", 0)) >= HEAVY_LIGAND_ATOM_THRESHOLD


def _attempt_seed(base_seed: int, seed_step: int, profile_index: int,
                  attempt: int, max_attempts: int) -> int:
    """Return a distinct deterministic seed for every profile/attempt pair."""
    offset = profile_index * max_attempts + (attempt - 1)
    return int(base_seed) + offset * int(seed_step)


def _run_isolated_child(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Run one complex in its own process group and kill the group on timeout."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            cmd, timeout, output=stdout or exc.output, stderr=stderr or exc.stderr
        ) from None
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def assemble_one(spec: pd.Series, args, known_bad: set[str]) -> dict:
    if getattr(args, "allow_simplified_ligand", False):
        raise RuntimeError(
            "Simplified ligand fallback is disabled; retry the original ligand instead."
        )

    out_root = Path(args.out)
    xyz_path, mol2_path, _ = expected_paths(spec, out_root)
    smiles = str(spec["SMILES_FOR_ARCHITECTOR"])
    coord = _coordlist(spec["COORDLIST"])
    ligtype_override = _ligtype_override_for_spec(spec, args)

    # 1. Already on disk -> never recompute, except for legacy rows that used a
    # chemically simplified ligand at the same expected path.
    force_rebuild_ids = getattr(args, "force_rebuild_build_ids", set())
    force_rebuild = str(spec["build_id"]) in force_rebuild_ids
    if args.skip_existing and not force_rebuild and valid_xyz(xyz_path):
        qc_status, qc_note = qc_xyz(xyz_path, spec)
        status = "existing_ok" if qc_status == "accepted" else "failed_qc"
        note = "found_valid_xyz" if status == "existing_ok" else f"found_valid_xyz; {qc_note}"
        return index_row(spec, status=status, note=note,
                         qc_status=qc_status, xyz_path=str(xyz_path),
                         mol2_path=str(mol2_path) if mol2_path.exists() else "")

    # 2. Ligand already known to fail ligType assignment.
    if _bad_key(smiles, spec["COORDLIST"]) in known_bad and not args.ignore_known_bad and not ligtype_override:
        return index_row(spec, status="skipped_known_bad_ligtype", note="cached_ligtype_failure")

    # 3. Cheap structural validation before the expensive build.
    ok, note = validate_coordlist(smiles, coord, spec.get("DENTATE"))
    if not ok:
        return index_row(spec, status="skipped_invalid_coordlist", note=note)

    # Auto-escalate bulky ligands to the hard-ligand profile sequence.
    profile_sequence = getattr(args, "profile_sequence", DEFAULT_PROFILE_SEQUENCE)
    if profile_sequence == DEFAULT_PROFILE_SEQUENCE and args.auto_hard_ligand and _heavy_ligand(spec):
        profile_sequence = HARD_LIGAND_PROFILE_SEQUENCE
    profiles = _profile_names(profile_sequence)
    max_attempts = max(1, int(args.max_attempts))
    max_total_seconds = max(0, int(getattr(args, "max_total_seconds", 0) or 0))
    started_at = time.monotonic()

    def run_profile_sequence(build_spec, *, simplified, success_note):
        last = None
        build_smiles = str(build_spec["SMILES_FOR_ARCHITECTOR"])
        build_coordlist = str(build_spec["COORDLIST"])
        build_ligtype_override = _ligtype_override_for_spec(build_spec, args)
        deadline_exhausted = False
        for profile_index, profile in enumerate(profiles):
            for attempt in range(1, max_attempts + 1):
                label = f"{profile} attempt {attempt}"
                attempt_timeout = int(args.timeout_per_complex)
                if max_total_seconds:
                    remaining = int(max_total_seconds - (time.monotonic() - started_at))
                    if remaining < 60:
                        note = f"generation budget exhausted before {label}"
                        last = (EXIT_WALLTIME_BUDGET, note, None, False, "")
                        deadline_exhausted = True
                        print(f"      {note}", flush=True)
                        break
                    attempt_timeout = min(attempt_timeout, remaining)
                attempt_seed = _attempt_seed(
                    args.seed, args.seed_step, profile_index, attempt, max_attempts
                )
                cmd = [
                    sys.executable, str(Path(__file__).resolve()), "--build-one",
                    "--specs", str(args.specs), "--build-id", str(spec["build_id"]),
                    "--out", str(out_root), "--n-symmetries", str(args.n_symmetries),
                    "--n-conformers", str(args.n_conformers),
                    "--xtb-max-iterations", str(args.xtb_max_iterations),
                    "--seed", str(attempt_seed), "--profile", profile,
                ]
                if simplified:
                    cmd.extend(["--override-smiles", build_smiles,
                                "--override-coordlist", build_coordlist])
                if build_ligtype_override:
                    cmd.extend(["--override-ligtype", build_ligtype_override])
                try:
                    proc = _run_isolated_child(cmd, timeout=attempt_timeout)
                except subprocess.TimeoutExpired:
                    xyz_path.with_suffix(".tmp.xyz").unlink(missing_ok=True)
                    note = f"{profile}: exceeded {attempt_timeout}s (attempt {attempt}, seed {attempt_seed})"
                    last = (124, note, None, False, "")
                    print(f"      {note}", flush=True)
                    break

                note, energy = _result_note(proc.stdout, proc.stderr)
                rc = proc.returncode
                if rc != EXIT_OK:
                    xyz_path.with_suffix(".tmp.xyz").unlink(missing_ok=True)

                if rc == EXIT_OK and valid_xyz(xyz_path):
                    qc_status, qc_note = qc_xyz(xyz_path, spec)
                    if qc_status != "accepted":
                        # Build succeeded but QC rejected it -> failed_qc; keep file.
                        last = (EXIT_QC, f"{qc_status}: {qc_note}", proc, False, qc_status)
                        print(f"      QC failed: {qc_status} ({qc_note})", flush=True)
                        break
                    status = "ok_simplified_ligand" if simplified else "ok"
                    lig_note = f"; ligType={build_ligtype_override}" if build_ligtype_override else ""
                    note_text = success_note if simplified else f"built ({label}, seed {attempt_seed}{lig_note})"
                    return index_row(
                        spec, status=status, energy_eV=energy, note=note_text,
                        qc_status=qc_status, xyz_path=str(xyz_path),
                        mol2_path=str(mol2_path) if mol2_path.exists() else "",
                        smiles_used=build_smiles, simplified_ligand=simplified,
                        ligtype_override=build_ligtype_override,
                    ), (rc, note_text, proc, False, qc_status)
                if rc == EXIT_LIGTYPE:
                    lig_note = f" ligType={build_ligtype_override}" if build_ligtype_override else ""
                    last = (rc, f"{profile}{lig_note}: {note or 'cannot_assign_ligtype'}", proc, False, "")
                    break
                native = rc < 0 or rc in (134, 136, 139)  # SIGABRT/SIGFPE/SIGSEGV
                last = (rc, f"{profile}: {note}", proc, native, "")
                if (native or rc == EXIT_NO_STRUCTURE) and attempt < max_attempts:
                    print(f"      retry {attempt}/{max_attempts - 1} after "
                          f"{'native_crash' if native else 'no_structures'} [{profile}]", flush=True)
                    continue
                break
            if deadline_exhausted:
                break
            print(f"      profile {profile} did not produce an accepted xyz", flush=True)
        return None, last

    row, last = run_profile_sequence(spec, simplified=False, success_note="")
    if row is not None:
        return row

    rc, note, proc, native, qc_status = last
    # Any generated valid XYZ is a real output even when cheap QC rejects it.
    # Preserve and index it so physical completeness is visible independently
    # from scientific QC acceptance.
    if valid_xyz(xyz_path):
        qc_status_now, qc_note = qc_xyz(xyz_path, spec)
        return index_row(
            spec, status="failed_qc",
            note=f"generated_xyz_preserved; {qc_note}; final_attempt={note}",
            qc_status=qc_status_now or qc_status or "QC_FAILED",
            xyz_path=str(xyz_path),
            mol2_path=str(mol2_path) if mol2_path.exists() else "",
            ligtype_override=ligtype_override,
        )

    if rc == EXIT_LIGTYPE:
        if not ligtype_override:
            append_known_bad(smiles, spec["COORDLIST"])
            known_bad.add(_bad_key(smiles, spec["COORDLIST"]))
        return index_row(
            spec, status="failed_ligtype", note=note or "cannot_assign_ligtype",
            ligtype_override=ligtype_override,
        )
    if rc == EXIT_NO_STRUCTURE:
        return index_row(
            spec, status="failed_no_structures",
            note=note or "no_structures_returned (repeated)",
            ligtype_override=ligtype_override,
        )
    if rc == 124:
        return index_row(spec, status="failed_timeout", note=note, ligtype_override=ligtype_override)
    if rc == EXIT_WALLTIME_BUDGET:
        return index_row(spec, status="failed_walltime_budget", note=note,
                         ligtype_override=ligtype_override)
    if native:
        if proc is not None:
            _save_failure_log(spec, proc)
        status = "failed_native_crash_repeated" if max_attempts > 1 else "failed_native_crash"
        return index_row(spec, status=status, note=f"exit{rc}: native crash", ligtype_override=ligtype_override)
    if proc is not None:
        _save_failure_log(spec, proc)
    return index_row(spec, status="failed_exception", note=note or f"exit{rc}", ligtype_override=ligtype_override)


def _save_failure_log(spec: pd.Series, proc) -> None:
    try:
        FAILURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = FAILURE_LOG_DIR / f"{spec['build_id']}.log"
        header = (
            f"# build_id={spec['build_id']}  metal={spec['metal_symbol']}  "
            f"Z={spec.get('Atomic Number_metal')}  ox={spec.get('metal_ox', 3)}\n"
            f"# coreCN={spec.get('coreCN')}  denticity={spec.get('DENTATE')}  "
            f"n_ligs={spec.get('n_ligs')}  anion={spec.get('inner_sphere_anion')}\n"
            f"# coordList={spec.get('COORDLIST')}  donors={spec.get('DONOR_TYPES')}\n"
            f"# smiles={spec['SMILES_FOR_ARCHITECTOR']}\n"
            f"# returncode={getattr(proc, 'returncode', '?')}\n"
        )
        log.write_text(f"{header}## STDOUT\n{proc.stdout}\n## STDERR\n{proc.stderr}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Targeted regeneration of nearest-coreCN FAIL_LONG_BOND geometries
# ---------------------------------------------------------------------------
def _first_present(row: pd.Series, names: list[str]):
    for name in names:
        if name not in row.index:
            continue
        value = row[name]
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if str(value).lower() == "nan":
            continue
        return value
    return None


def _spec_lookup(specs: pd.DataFrame) -> dict[str, pd.Series]:
    if "build_id" not in specs.columns:
        raise SystemExit(f"{DEFAULT_SPECS} is missing build_id")
    return {str(r["build_id"]): r for _, r in specs.iterrows()}


def _identity_value(name: str, value):
    if name == "COORDLIST":
        return tuple(_coordlist(value))
    if name in {"DENTATE", "coreCN", "n_ligs", "n_fill", "Atomic Number_metal", "metal_ox"}:
        return _safe_int(value, -1)
    return _safe_text(value).strip()


def _regeneration_spec(row: pd.Series, specs_by_id: dict[str, pd.Series]) -> tuple[pd.Series | None, str]:
    build_id = _first_present(row, ["build_id", "geometry_build_id"])
    if build_id is None:
        return None, "missing_build_id"
    build_id = str(build_id)
    frozen = specs_by_id.get(build_id)
    spec = (frozen if frozen is not None else row).copy()
    spec["build_id"] = build_id

    overrides = {
        "SMILES_FOR_ARCHITECTOR": [
            "smiles_for_architector_used", "SMILES_FOR_ARCHITECTOR",
            "smiles_for_architector_original", "canonical_smiles",
        ],
        "COORDLIST": ["COORDLIST", "coordList", "coordlist"],
        "DENTATE": ["DENTATE", "dentate"],
        "coreCN": ["coreCN", "core_cn"],
        "n_ligs": ["n_ligs", "n_ligands"],
        "inner_sphere_anion": ["inner_sphere_anion", "fill_ligand"],
        "metal_symbol": ["metal_symbol", "metal"],
        "Atomic Number_metal": ["Atomic Number_metal", "atomic_number_metal", "Z"],
        "metal_ox": ["metal_ox", "oxidation_state", "metal_oxidation_state"],
        "geometry_key": ["geometry_key"],
        "DONOR_TYPES": ["DONOR_TYPES", "donor_types"],
        "fill_ligand": ["fill_ligand", "inner_sphere_anion"],
        "n_fill": ["n_fill"],
    }
    for target, names in overrides.items():
        value = _first_present(row, names)
        if value is not None:
            if frozen is not None and target in {
                "SMILES_FOR_ARCHITECTOR", "COORDLIST", "DENTATE", "coreCN",
                "n_ligs", "inner_sphere_anion", "fill_ligand", "n_fill",
            }:
                frozen_value = frozen.get(target, "")
                if _identity_value(target, value) != _identity_value(target, frozen_value):
                    return None, f"build_id_identity_mismatch:{target}"
            spec[target] = value

    for name in (
        "run_id", "queue_sha256", "strategy_sha256", "root_source_build_id",
        "source_build_id", "parent_build_id", "rescue_route", "hypothesis_version",
        "ligtype_sequence", "source_failure_class", "hypothesis_reason",
    ):
        value = _first_present(row, [name])
        if value is not None:
            spec[name] = value

    if _safe_text(spec.get("metal_symbol", "")) and not _safe_text(spec.get("Atomic Number_metal", "")):
        spec["Atomic Number_metal"] = Z_BY_LANTHANIDE_SYMBOL.get(_safe_text(spec["metal_symbol"]), "")
    if not _safe_text(spec.get("metal_ox", "")):
        spec["metal_ox"] = 3
    if not _safe_text(spec.get("fill_ligand", "")):
        spec["fill_ligand"] = spec.get("inner_sphere_anion", "water")
    if not _safe_text(spec.get("n_fill", "")):
        spec["n_fill"] = ""

    required = [
        "SMILES_FOR_ARCHITECTOR", "COORDLIST", "metal_symbol", "Atomic Number_metal",
        "metal_ox", "coreCN", "n_ligs", "inner_sphere_anion",
    ]
    missing = [name for name in required if not _safe_text(spec.get(name, ""))]
    if missing:
        return None, f"missing_required_columns:{','.join(missing)}"
    if not _coordlist(spec["COORDLIST"]):
        return None, "missing_or_invalid_fixed_coordlist"
    coord = _coordlist(spec["COORDLIST"])
    denticity = _safe_int(spec.get("DENTATE"), 0)
    core_cn = _safe_int(spec.get("coreCN"), 0)
    n_ligs = _safe_int(spec.get("n_ligs"), 0)
    if denticity != len(coord):
        return None, "coordlist_denticity_mismatch"
    valid_coord, coord_note = validate_coordlist(
        _safe_text(spec.get("SMILES_FOR_ARCHITECTOR")), coord, denticity,
    )
    if not valid_coord:
        return None, f"invalid_fixed_coordlist:{coord_note}"
    if core_cn < denticity * n_ligs:
        return None, "coreCN_smaller_than_ligand_donor_capacity"
    if _safe_text(spec.get("n_fill", "")):
        n_fill = _safe_int(spec.get("n_fill"), -1)
        if n_fill != core_cn - denticity * n_ligs:
            return None, "n_fill_coordination_balance_mismatch"

    if frozen is None:
        if not _safe_text(spec.get("n_fill", "")):
            return None, "missing_required_columns:n_fill"
        expected_build_id = complex_build_id(
            metal_Z=_safe_int(spec.get("Atomic Number_metal"), 0),
            ligand_smiles=_safe_text(spec.get("SMILES_FOR_ARCHITECTOR")),
            coord_list=coord,
            denticity=denticity,
            core_cn=core_cn,
            n_ligs=n_ligs,
            inner_sphere_anion=_safe_text(spec.get("inner_sphere_anion")),
            fill_ligand=_safe_text(spec.get("fill_ligand"), _safe_text(spec.get("inner_sphere_anion"))),
            n_fill_value=_safe_int(spec.get("n_fill"), 0),
        )
        if expected_build_id != build_id:
            return None, f"new_hypothesis_build_id_mismatch:expected={expected_build_id}"

    root_source = _safe_text(spec.get("root_source_build_id")) or _safe_text(
        spec.get("source_build_id")
    ) or build_id
    spec["root_source_build_id"] = root_source
    if not _safe_text(spec.get("source_build_id")):
        spec["source_build_id"] = root_source
    if not _safe_text(spec.get("parent_build_id")):
        spec["parent_build_id"] = build_id
    return spec, "ok"


def _source_xyz_path(row: pd.Series, spec: pd.Series) -> str:
    value = _first_present(row, ["xyz_path", "source_xyz_path", "old_xyz_path"])
    if value is not None:
        return str(value)
    xyz_path, _, _ = expected_paths(spec, GEOMETRY_DIR)
    return str(xyz_path)


def _stable_seed(build_id: str, base_seed: int, attempt: int, seed_step: int) -> int:
    digest = hashlib.sha256(str(build_id).encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return (int(base_seed) + offset + attempt * int(seed_step)) % (2**31 - 1)


def _jsonable_spec(spec: pd.Series) -> dict:
    payload = {}
    for key in CHILD_SPEC_FIELDS:
        value = spec.get(key, "")
        try:
            if pd.isna(value):
                payload[key] = ""
                continue
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        payload[key] = value
    return payload


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_dup{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could_not_choose_unique_destination_for_{path}")


def _store_candidate(
    work_xyz: Path,
    work_mol2: Path,
    bucket_root: Path,
    *,
    attempt: int,
    profile: str,
) -> tuple[str, str]:
    safe_profile = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile)
    dest_dir = bucket_root / work_xyz.parent.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_attempt{attempt:02d}_{safe_profile}"
    dest_xyz = _unique_destination(dest_dir / f"{work_xyz.stem}{suffix}{work_xyz.suffix}")
    shutil.move(str(work_xyz), dest_xyz)
    dest_mol2 = ""
    if work_mol2.exists():
        mol2_target = _unique_destination(dest_dir / f"{work_mol2.stem}{suffix}{work_mol2.suffix}")
        shutil.move(str(work_mol2), mol2_target)
        dest_mol2 = str(mol2_target)
    return str(dest_xyz), dest_mol2


def _copy_best_failed(row: dict, best_failed_root: Path) -> str:
    src = Path(str(row.get("generated_xyz_path", "")))
    if not src.exists():
        return ""
    dest_dir = best_failed_root / src.parent.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_destination(dest_dir / src.name)
    shutil.copy2(src, dest)
    mol2 = Path(str(row.get("generated_mol2_path", "")))
    if mol2.exists():
        shutil.copy2(mol2, _unique_destination(dest_dir / mol2.name))
    return str(dest)


def _is_fail_qc_class(qc_class: str) -> bool:
    text = str(qc_class)
    return text.startswith("FAIL") or text in {"QC_FAILED", "BUILD_FAILED", "NO_XYZ"}


def _best_attempt_rank(row: dict) -> tuple[int, float, float, int]:
    qc_class = _safe_text(row.get("qc_class", ""))
    class_rank = {
        "OK": 0,
        "BORDERLINE_LONGISH": 1,
        "BORDERLINE_AMBIGUOUS_SHELL": 2,
        "FAIL_LONG_BOND": 3,
        "NITRATE_MISSING": 4,
        "WATER_MISSING": 4,
        "COMPOSITION_MISMATCH": 4,
        "FAIL_TOO_FEW_DONORS": 5,
        "FAIL_NO_METAL": 6,
        "QC_FAILED": 7,
        "BUILD_FAILED": 8,
    }.get(qc_class, 7)
    max_dist = _safe_float(row.get("coreCN_max_dist"), math.inf)
    gap = _safe_float(row.get("gap_after_coreCN"), -math.inf)
    gap_rank = -gap if not math.isnan(gap) else math.inf
    return (
        class_rank,
        max_dist if not math.isnan(max_dist) else math.inf,
        gap_rank,
        _safe_int(row.get("attempt"), 0),
    )


def _attempt_base_row(
    queue_index: int,
    spec: pd.Series,
    source_xyz: str,
    attempt: int,
    profile: str,
    seed: int,
    n_symmetries: int,
    n_conformers: int,
    args,
) -> dict:
    return {
        "queue_index": queue_index,
        "build_id": str(spec["build_id"]),
        **{
            name: _safe_text(spec.get(name, getattr(args, name, "")))
            for name in REGEN_PROVENANCE_FIELDS
        },
        "metal_symbol": _safe_text(spec.get("metal_symbol", "")),
        "Atomic Number_metal": _safe_text(spec.get("Atomic Number_metal", "")),
        "metal_ox": _safe_text(spec.get("metal_ox", "")),
        "smiles_for_architector_used": _safe_text(spec.get("SMILES_FOR_ARCHITECTOR", "")),
        "COORDLIST": _safe_text(spec.get("COORDLIST", "")),
        "DENTATE": _safe_text(spec.get("DENTATE", "")),
        "coreCN": _safe_text(spec.get("coreCN", "")),
        "n_ligs": _safe_text(spec.get("n_ligs", "")),
        "inner_sphere_anion": _safe_text(spec.get("inner_sphere_anion", "")),
        "fill_ligand": _safe_text(spec.get("fill_ligand", "")),
        "n_fill": _safe_text(spec.get("n_fill", "")),
        "geometry_key": _safe_text(spec.get("geometry_key", "")),
        "source_xyz_path": source_xyz,
        "attempt": attempt,
        "profile": profile,
        "seed": seed,
        "n_symmetries": n_symmetries,
        "n_conformers": n_conformers,
        "xtb_max_iterations": int(args.xtb_max_iterations),
        "returncode": "",
        "relax": "",
        "assemble_method": "",
        "full_method": "",
        "force_generation": "",
        "ff_preopt": "",
        "attempt_status": "",
        "accepted_for_clean_3d_features": False,
        "ligtype_override": "",
        "generated_xyz_path": "",
        "generated_mol2_path": "",
        "energy_eV": "",
        "note": "",
        "file_qc_status": "",
        "file_qc_note": "",
        "qc_class": "",
        "qc_note": "",
        "nearest_coreCN_sig": "",
        "coreCN_max_dist": "",
        "next_donor_dist": "",
        "gap_after_coreCN": "",
        "all_nearest": "",
    }


def _run_regeneration_attempt(
    queue_index: int,
    spec: pd.Series,
    source_xyz: str,
    attempt: int,
    profile: str,
    out_root: Path,
    args,
) -> dict:
    base_seed = _stable_seed(str(spec["build_id"]), int(args.seed), attempt, int(args.seed_step))
    requested_symmetries = max(
        1, int(args.n_symmetries) + (attempt - 1) * int(args.n_symmetries_step)
    )
    requested_conformers = max(
        1, int(args.n_conformers) + (attempt - 1) * int(args.n_conformers_step)
    )
    profile_settings = BUILD_PROFILES[profile]
    n_symmetries = int(_profile_value(profile_settings, "n_symmetries", requested_symmetries))
    n_conformers = int(_profile_value(profile_settings, "n_conformers", requested_conformers))
    xtb_max_iterations = int(
        _profile_value(profile_settings, "xtb_max_iterations", args.xtb_max_iterations)
    )
    effective_seed = int(base_seed) + int(profile_settings.get("seed_offset", 0))
    ligtype_candidates = _ligtype_candidates_for_spec(spec, args)
    ligtype_choice = ligtype_candidates[(attempt - 1) % len(ligtype_candidates)]
    ligtype_override = "" if ligtype_choice == AUTO_LIGTYPE_TOKEN else ligtype_choice
    row = _attempt_base_row(
        queue_index, spec, source_xyz, attempt, profile, effective_seed,
        n_symmetries, n_conformers, args,
    )
    row["ligtype_override"] = ligtype_override
    row.update({
        "xtb_max_iterations": xtb_max_iterations,
        "relax": bool(profile_settings.get("relax", True)),
        "assemble_method": str(profile_settings.get("assemble_method", "GFN2-xTB")),
        "full_method": str(profile_settings.get("full_method", "GFN2-xTB")),
        "force_generation": bool(profile_settings.get("force_generation", False)),
        "ff_preopt": bool(profile_settings.get("ff_preopt", True)),
    })

    work_parent = out_root / "_work"
    work_parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(
        prefix=f"{spec['build_id']}_attempt{attempt:02d}_",
        dir=work_parent,
    ))
    work_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--build-one",
        "--build-id", str(spec["build_id"]), "--spec-json", json.dumps(_jsonable_spec(spec)),
        "--out", str(work_root), "--n-symmetries", str(n_symmetries),
        "--n-conformers", str(n_conformers),
        "--xtb-max-iterations", str(xtb_max_iterations),
        "--seed", str(base_seed), "--profile", profile,
    ]
    if ligtype_override:
        cmd.extend(["--override-ligtype", ligtype_override])

    proc = None
    try:
        proc = _run_isolated_child(cmd, timeout=int(args.timeout_per_complex))
    except subprocess.TimeoutExpired:
        row.update({
            "returncode": 124,
            "attempt_status": "failed_timeout",
            "note": f"exceeded {int(args.timeout_per_complex)}s",
            "qc_class": "BUILD_FAILED",
            "qc_note": "timeout",
        })
        shutil.rmtree(work_root, ignore_errors=True)
        return row

    note, energy = _result_note(proc.stdout, proc.stderr)
    row["returncode"] = proc.returncode
    row["energy_eV"] = energy
    row["note"] = note

    work_xyz, work_mol2, _ = expected_paths(spec, work_root)
    if proc.returncode != EXIT_OK or not valid_xyz(work_xyz):
        row["attempt_status"] = "build_failed"
        row["qc_class"] = "BUILD_FAILED"
        row["qc_note"] = note or f"exit{proc.returncode}"
        if proc.returncode < 0 or proc.returncode in (134, 136, 139):
            _save_failure_log(spec, proc)
        shutil.rmtree(work_root, ignore_errors=True)
        return row

    file_qc_status, file_qc_note = qc_xyz(work_xyz, spec)
    qc = nearest_corecn_qc_xyz(
        work_xyz,
        spec,
        long_bond_threshold=float(args.long_bond_threshold),
        borderline_longish_threshold=float(args.borderline_longish_threshold),
        ambiguous_gap_threshold=float(args.ambiguous_gap_threshold),
    )
    if file_qc_status != "accepted":
        qc["qc_class"] = file_qc_status
        qc["qc_note"] = file_qc_note
    accepted = file_qc_status == "accepted" and qc["qc_class"] == "OK"
    bucket = out_root / ("accepted" if accepted else "rejected")
    stored_xyz, stored_mol2 = _store_candidate(
        work_xyz, work_mol2, bucket, attempt=attempt, profile=profile,
    )
    shutil.rmtree(work_root, ignore_errors=True)

    row.update(qc)
    row.update({
        "file_qc_status": file_qc_status,
        "file_qc_note": file_qc_note,
        "attempt_status": "accepted" if accepted else "rejected_by_dual_qc",
        "accepted_for_clean_3d_features": bool(accepted),
        "generated_xyz_path": stored_xyz,
        "generated_mol2_path": stored_mol2,
    })
    return row


def _accepted_report_row(attempt_row: dict) -> dict:
    return {
        "queue_index": attempt_row["queue_index"],
        "build_id": attempt_row["build_id"],
        **{name: attempt_row.get(name, "") for name in REGEN_PROVENANCE_FIELDS},
        "metal_symbol": attempt_row["metal_symbol"],
        "Atomic Number_metal": attempt_row["Atomic Number_metal"],
        "metal_ox": attempt_row["metal_ox"],
        "smiles_for_architector_used": attempt_row["smiles_for_architector_used"],
        "COORDLIST": attempt_row["COORDLIST"],
        "DENTATE": attempt_row["DENTATE"],
        "coreCN": attempt_row["coreCN"],
        "n_ligs": attempt_row["n_ligs"],
        "inner_sphere_anion": attempt_row["inner_sphere_anion"],
        "fill_ligand": attempt_row.get("fill_ligand", ""),
        "n_fill": attempt_row.get("n_fill", ""),
        "geometry_key": attempt_row["geometry_key"],
        "accepted_attempt": attempt_row["attempt"],
        "accepted_profile": attempt_row.get("profile", ""),
        "accepted_ligtype_override": attempt_row.get("ligtype_override", ""),
        "relax": attempt_row.get("relax", ""),
        "assemble_method": attempt_row.get("assemble_method", ""),
        "full_method": attempt_row.get("full_method", ""),
        "force_generation": attempt_row.get("force_generation", ""),
        "ff_preopt": attempt_row.get("ff_preopt", ""),
        gschema.ACCEPTED_XYZ_PATH: attempt_row["generated_xyz_path"],
        gschema.ACCEPTED_MOL2_PATH: attempt_row["generated_mol2_path"],
        gschema.QC_CLASS: attempt_row["qc_class"],
        gschema.CORECN_MAX_DIST: attempt_row["coreCN_max_dist"],
        gschema.GAP_AFTER_CORECN: attempt_row["gap_after_coreCN"],
        "nearest_coreCN_sig": attempt_row["nearest_coreCN_sig"],
        "file_qc_status": attempt_row.get("file_qc_status", ""),
        "file_qc_note": attempt_row.get("file_qc_note", ""),
        "accepted_for_clean_3d_features": True,
        "ligtype_override": attempt_row.get("ligtype_override", ""),
    }


def _still_failed_report_row(
    queue_index: int,
    spec: pd.Series,
    attempts: list[dict],
    best_failed_root: Path,
    failure_note: str,
) -> dict:
    generated = [row for row in attempts if row.get("generated_xyz_path")]
    best = min(generated, key=_best_attempt_rank) if generated else None
    best_failed_xyz = _copy_best_failed(best, best_failed_root) if best else ""
    attempts_run = len(attempts)
    ligtype_override = next((row.get("ligtype_override", "") for row in attempts if row.get("ligtype_override")), "")
    return {
        "queue_index": queue_index,
        "build_id": str(spec.get("build_id", "")),
        **{name: _safe_text(spec.get(name, "")) for name in REGEN_PROVENANCE_FIELDS},
        "metal_symbol": _safe_text(spec.get("metal_symbol", "")),
        "Atomic Number_metal": _safe_text(spec.get("Atomic Number_metal", "")),
        "metal_ox": _safe_text(spec.get("metal_ox", "")),
        "smiles_for_architector_used": _safe_text(spec.get("SMILES_FOR_ARCHITECTOR", "")),
        "COORDLIST": _safe_text(spec.get("COORDLIST", "")),
        "DENTATE": _safe_text(spec.get("DENTATE", "")),
        "coreCN": _safe_text(spec.get("coreCN", "")),
        "n_ligs": _safe_text(spec.get("n_ligs", "")),
        "inner_sphere_anion": _safe_text(spec.get("inner_sphere_anion", "")),
        "fill_ligand": _safe_text(spec.get("fill_ligand", "")),
        "n_fill": _safe_text(spec.get("n_fill", "")),
        "geometry_key": _safe_text(spec.get("geometry_key", "")),
        "attempts_run": attempts_run,
        "best_qc_class": best.get("qc_class", "") if best else "",
        "best_file_qc_status": best.get("file_qc_status", "") if best else "",
        "best_coreCN_max_dist": best.get("coreCN_max_dist", "") if best else "",
        "best_gap_after_coreCN": best.get("gap_after_coreCN", "") if best else "",
        "best_xyz_path": best.get("generated_xyz_path", "") if best else "",
        "best_failed_xyz_path": best_failed_xyz,
        "failure_note": failure_note,
        "accepted_for_clean_3d_features": False,
        "ligtype_override": ligtype_override,
    }


def _write_regeneration_summary(
    path: Path,
    *,
    args,
    input_rows: int,
    queued_rows: int,
    attempt_rows: list[dict],
    accepted_rows: list[dict],
    still_failed_rows: list[dict],
    missing_shard_reports: int = 0,
) -> dict[str, int]:
    def queue_indexes(rows: list[dict]) -> set[str]:
        indexes = set()
        for row in rows:
            value = _safe_text(row.get("queue_index", "")).strip()
            if value and value.lower() != "nan":
                indexes.add(value)
        return indexes

    started_indexes = (
        queue_indexes(attempt_rows)
        | queue_indexes(accepted_rows)
        | queue_indexes(still_failed_rows)
    )
    completed_indexes = queue_indexes(accepted_rows) | queue_indexes(still_failed_rows)
    incomplete_rows = max(int(queued_rows) - len(completed_indexes), 0)
    attempts_by_class = {}
    for row in attempt_rows:
        key = row.get("qc_class", "")
        attempts_by_class[key] = attempts_by_class.get(key, 0) + 1
    lines = [
        "Regenerated FAIL_LONG_BOND geometry summary",
        "",
        f"Input file: {args.regenerate_input}",
        f"Rows in input: {input_rows}",
        f"Rows queued for FAIL_LONG_BOND regeneration: {queued_rows}",
        f"Rows started: {len(started_indexes)}",
        f"Rows completed: {len(completed_indexes)}",
        f"Rows incomplete: {incomplete_rows}",
        f"Missing shard reports: {int(missing_shard_reports)}",
        f"Attempts recorded: {len(attempt_rows)}",
        f"Accepted clean regenerated geometries: {len(accepted_rows)}",
        f"Still not accepted for clean 3D features: {len(still_failed_rows)}",
        "",
        "Nearest-coreCN QC thresholds:",
        f"  FAIL_LONG_BOND if coreCN_max_dist > {float(args.long_bond_threshold):.2f} A",
        f"  BORDERLINE_AMBIGUOUS_SHELL if gap_after_coreCN < {float(args.ambiguous_gap_threshold):.2f} A",
        f"  BORDERLINE_LONGISH if coreCN_max_dist > {float(args.borderline_longish_threshold):.2f} A",
        "",
        "Attempt QC class counts:",
        json.dumps(attempts_by_class, indent=2, sort_keys=True),
        "",
        f"Regenerated geometry root: {args.regen_out}",
        "Accepted candidates: accepted/",
        "Rejected candidates: rejected/",
        "Best non-OK candidates: best_failed/",
        "",
        "Original data/geometries files were not overwritten.",
        "Only qc_class == OK rows in regenerated_fail_long_bond_accepted.csv are clean feature candidates.",
    ]
    if incomplete_rows or missing_shard_reports:
        lines.extend([
            "",
            "WARNING: regeneration is incomplete; partial reports were preserved for resume.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "started_rows": len(started_indexes),
        "completed_rows": len(completed_indexes),
        "incomplete_rows": incomplete_rows,
        "missing_shard_reports": int(missing_shard_reports),
    }


def _regen_shard_suffix(args) -> str:
    num_shards = int(getattr(args, "num_shards", 1))
    if num_shards <= 1:
        return ""
    shard_id = int(getattr(args, "shard_id", 0))
    return f"shard{shard_id}of{num_shards}"


def _regen_report_path(reports_dir: Path, name: str, args) -> Path:
    suffix = _regen_shard_suffix(args)
    if not suffix:
        return reports_dir / name
    path = Path(name)
    return reports_dir / f"{path.stem}_{suffix}{path.suffix}"


def _sha256_file(path: Path | None) -> str:
    if path is None or not Path(path).exists():
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regeneration_strategy_sha256(args) -> str:
    implementation_files = (
        Path(__file__).resolve(),
        _REPO_ROOT / "src" / "chemistry" / "coordination.py",
        _REPO_ROOT / "src" / "geometry_schema.py",
    )
    payload = {
        "run_id": _safe_text(getattr(args, "run_id", "")),
        "queue_sha256": _safe_text(getattr(args, "queue_sha256", "")),
        "num_shards": int(getattr(args, "num_shards", 1)),
        "max_attempts": int(getattr(args, "max_attempts", 2)),
        "timeout_per_complex": int(getattr(args, "timeout_per_complex", 1800)),
        "seed": int(getattr(args, "seed", 0xF00D)),
        "seed_step": int(getattr(args, "seed_step", 7919)),
        "profile_sequence": list(_profile_names(getattr(args, "profile_sequence", "standard"))),
        "n_symmetries": int(getattr(args, "n_symmetries", 40)),
        "n_symmetries_step": int(getattr(args, "n_symmetries_step", 10)),
        "n_conformers": int(getattr(args, "n_conformers", 5)),
        "n_conformers_step": int(getattr(args, "n_conformers_step", 1)),
        "xtb_max_iterations": int(getattr(args, "xtb_max_iterations", 250)),
        "long_bond_threshold": float(getattr(args, "long_bond_threshold", DEFAULT_LONG_BOND_THRESHOLD)),
        "borderline_longish_threshold": float(getattr(
            args, "borderline_longish_threshold", DEFAULT_BORDERLINE_LONGISH_THRESHOLD,
        )),
        "ambiguous_gap_threshold": float(getattr(
            args, "ambiguous_gap_threshold", DEFAULT_AMBIGUOUS_GAP_THRESHOLD,
        )),
        "ligtype_overrides_sha256": _sha256_file(getattr(args, "ligtype_overrides", None)),
        "implementation_sha256": {
            _portable_path(path): _sha256_file(path) for path in implementation_files
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rows_for_regeneration_run(frame: pd.DataFrame, args) -> pd.DataFrame:
    run_id = _safe_text(getattr(args, "run_id", ""))
    if not run_id or frame.empty:
        return frame
    required = {"run_id", "queue_sha256", "strategy_sha256"}
    if not required.issubset(frame.columns):
        return frame.iloc[0:0].copy()
    return frame[
        frame["run_id"].astype(str).eq(run_id)
        & frame["queue_sha256"].astype(str).eq(_safe_text(args.queue_sha256))
        & frame["strategy_sha256"].astype(str).eq(_safe_text(args.strategy_sha256))
    ].copy()


def _write_regeneration_run_meta(path: Path, args) -> None:
    payload = {
        "run_id": _safe_text(args.run_id),
        "queue_sha256": _safe_text(args.queue_sha256),
        "strategy_sha256": _safe_text(args.strategy_sha256),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(payload, path)


def _write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def _run_meta_matches(path: Path, args, expected_shard_id: int | None = None) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    shard_matches = (
        True if expected_shard_id is None
        else _safe_int(payload.get("shard_id"), -1) == int(expected_shard_id)
    )
    return all((
        _safe_text(payload.get("run_id")) == _safe_text(args.run_id),
        _safe_text(payload.get("queue_sha256")) == _safe_text(args.queue_sha256),
        _safe_text(payload.get("strategy_sha256")) == _safe_text(args.strategy_sha256),
        _safe_int(payload.get("num_shards"), -1) == int(args.num_shards),
        shard_matches,
    ))


def _read_regen_shard_reports(reports_dir: Path, name: str, num_shards: int) -> pd.DataFrame:
    if int(num_shards) <= 1:
        # A one-shard worker deliberately uses the canonical report name (no
        # ``_shard0of1`` suffix).  The merge must follow the same convention.
        return _read_csv_if_exists(reports_dir / name)

    stem = Path(name).stem
    suffix = Path(name).suffix
    frames = []
    for shard_id in range(int(num_shards)):
        path = reports_dir / f"{stem}_shard{shard_id}of{int(num_shards)}{suffix}"
        frame = _read_csv_if_exists(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _read_all_regen_shard_reports(reports_dir: Path, name: str) -> pd.DataFrame:
    """Read a report across shard cardinalities from resumed/pilot runs.

    A pilot may use (for example) 8 shards and the full run 16.  Accepted rows
    are durable work and must remain discoverable after that cardinality
    change; matching the complete filename prevents unrelated CSVs from being
    folded into the run.
    """
    stem = re.escape(Path(name).stem)
    suffix = re.escape(Path(name).suffix)
    pattern = re.compile(rf"{stem}_shard\d+of\d+{suffix}")
    frames = []
    for path in sorted(reports_dir.glob(f"{Path(name).stem}_shard*of*{Path(name).suffix}")):
        if pattern.fullmatch(path.name):
            frame = _read_csv_if_exists(path)
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _deduplicate_regeneration_attempts(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove canonical/shard copies of the same physical attempt.

    Strict runs have an immutable run/queue/strategy identity.  Legacy reports
    fall back to profile and seed so independent reruns remain distinguishable
    when that provenance exists.  Duplicate report copies must never count as
    repeated scientific evidence for an adaptive coordination hypothesis.
    """
    if frame.empty or not {"build_id", "attempt"}.issubset(frame.columns):
        return frame
    seen: set[tuple] = set()
    keep: list[int] = []
    for index, row in frame.iterrows():
        run_id = _safe_text(row.get("run_id"))
        if run_id:
            key = (
                "strict", run_id, _safe_text(row.get("queue_sha256")),
                _safe_text(row.get("strategy_sha256")),
                _safe_text(row.get("build_id")), _safe_int(row.get("attempt"), -1),
            )
        else:
            key = (
                "legacy", _safe_text(row.get("build_id")),
                _safe_int(row.get("attempt"), -1), _safe_text(row.get("profile")),
                _safe_text(row.get("seed")),
            )
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    return frame.loc[keep].copy()


def _regen_shard_report_paths(reports_dir: Path, name: str, num_shards: int) -> list[Path]:
    if int(num_shards) <= 1:
        return [reports_dir / name]
    stem = Path(name).stem
    suffix = Path(name).suffix
    return [
        reports_dir / f"{stem}_shard{shard_id}of{int(num_shards)}{suffix}"
        for shard_id in range(int(num_shards))
    ]


def _ensure_csv(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        _reset_csv(path, fields)
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        existing_header = next(csv.reader(handle), [])
    if existing_header == fields:
        return

    # Regeneration reports are append-only resume state.  New provenance/QC
    # columns must be added atomically before a new row is appended; otherwise
    # a wider DictWriter row under a historical header corrupts the CSV.
    frame = _read_csv_if_exists(path)
    unknown = sorted(set(frame.columns) - set(fields))
    if unknown:
        raise SystemExit(
            f"Refusing to drop unknown columns while upgrading {path}: {unknown}"
        )
    for field in fields:
        if field not in frame.columns:
            frame[field] = ""
    _write_csv_atomic(frame.loc[:, fields], path)


def _reset_csv(path: Path, fields: list[str]) -> None:
    """Start a fresh per-shard report while preserving accepted resume state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def _append_csv_row(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        # pandas writes a single newline for a DataFrame that has neither rows
        # nor columns.  Historical zero-acceptance merges therefore contain a
        # real file that is still semantically an empty report.
        return pd.DataFrame()


def _truthy_csv_value(value) -> bool:
    text = _safe_text(value, "").strip().lower()
    return text not in {"", "0", "false", "no", "nan"}


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _accepted_report_rows_for_resume(reports_dir: Path, out_root: Path, args) -> dict[str, dict]:
    """Current-run accepted rows that are safe to treat as completed work."""
    frames = []
    canonical = reports_dir / REGENERATE_ACCEPTED_NAME
    if canonical.exists():
        frames.append(_read_csv_if_exists(canonical))
    shard_rows = _read_all_regen_shard_reports(reports_dir, REGENERATE_ACCEPTED_NAME)
    if not shard_rows.empty:
        frames.append(shard_rows)
    if not frames:
        return {}

    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows = _rows_for_regeneration_run(rows, args)
    if rows.empty or gschema.BUILD_ID not in rows.columns:
        return {}

    accepted: dict[str, dict] = {}
    for row in rows.to_dict("records"):
        if _safe_text(row.get(gschema.QC_CLASS, "")) != "OK":
            continue
        if not _truthy_csv_value(row.get("accepted_for_clean_3d_features", True)):
            continue
        xyz = Path(_safe_text(row.get(gschema.ACCEPTED_XYZ_PATH, "")))
        if not valid_xyz(xyz) or not _path_is_under(xyz, out_root):
            continue
        build_id = _safe_text(row.get(gschema.BUILD_ID, ""))
        if build_id and build_id not in accepted:
            accepted[build_id] = row
    return accepted


def _regeneration_queue_rows(queue: pd.DataFrame) -> pd.DataFrame:
    """Select rows accepted by the focused regeneration worker.

    Legacy queues use ``qc_class=FAIL_LONG_BOND``.  Diagnostic hypothesis
    queues retain the real source QC class and identify their explicit rescue
    route instead of relabelling a borderline geometry as a long-bond failure.
    """
    if "qc_class" not in queue.columns:
        return queue.reset_index(drop=True)
    selected = queue["qc_class"].astype(str).eq("FAIL_LONG_BOND")
    if "rescue_route" in queue.columns:
        selected |= queue["rescue_route"].astype(str).isin(SUPPORTED_REGENERATION_ROUTES)
    return queue.loc[selected].reset_index(drop=True)


def _repeated_ambiguous_shell(attempts: list[dict], long_bond_threshold: float) -> bool:
    """Stop seed/profile retries after the same non-long-bond diagnosis twice."""
    if len(attempts) < 2:
        return False
    recent = attempts[-2:]
    if any(row.get("qc_class") != "BORDERLINE_AMBIGUOUS_SHELL" for row in recent):
        return False
    distances = [_safe_float(row.get("coreCN_max_dist")) for row in recent]
    return all(math.isfinite(value) and value <= long_bond_threshold for value in distances)


def _accepted_xyz_for_spec(spec: pd.Series, out_root: Path) -> Path | None:
    build_id = str(spec["build_id"])
    symbol = _safe_text(spec.get("metal_symbol", ""))
    roots = [out_root / "accepted" / symbol, out_root / "accepted"] if symbol else [out_root / "accepted"]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(f"*{build_id}*.xyz")):
            if path in seen:
                continue
            seen.add(path)
            if valid_xyz(path):
                return path
    return None


def _accepted_report_row_from_existing(queue_index: int, spec: pd.Series, xyz_path: Path, args) -> dict | None:
    file_qc_status, file_qc_note = qc_xyz(xyz_path, spec)
    if file_qc_status != "accepted":
        return None
    qc = nearest_corecn_qc_xyz(
        xyz_path,
        spec,
        long_bond_threshold=float(args.long_bond_threshold),
        borderline_longish_threshold=float(args.borderline_longish_threshold),
        ambiguous_gap_threshold=float(args.ambiguous_gap_threshold),
    )
    if qc.get("qc_class") != "OK":
        return None
    mol2_path = xyz_path.with_suffix(".mol2")
    attempt_row = _attempt_base_row(
        queue_index=queue_index,
        spec=spec,
        source_xyz=str(xyz_path),
        attempt=0,
        profile="existing_accepted",
        seed="",
        n_symmetries="",
        n_conformers="",
        args=args,
    )
    attempt_row.update(qc)
    attempt_row.update({
        "attempt_status": "existing_accepted",
        "accepted_for_clean_3d_features": True,
        "generated_xyz_path": str(xyz_path),
        "generated_mol2_path": str(mol2_path) if mol2_path.exists() else "",
        "note": "resume_existing_accepted_xyz",
        "file_qc_status": file_qc_status,
        "file_qc_note": file_qc_note,
    })
    return _accepted_report_row(attempt_row)


def _read_shard_outputs(
    attempts_file: Path,
    accepted_file: Path,
    still_failed_file: Path,
    args=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    attempts = _read_csv_if_exists(attempts_file)
    accepted = _read_csv_if_exists(accepted_file)
    still_failed = _read_csv_if_exists(still_failed_file)
    if args is not None:
        attempts = _rows_for_regeneration_run(attempts, args)
        accepted = _rows_for_regeneration_run(accepted, args)
        still_failed = _rows_for_regeneration_run(still_failed, args)
    accepted_build_ids = set()
    if not accepted.empty and gschema.BUILD_ID in accepted.columns:
        ok_mask = (
            accepted[gschema.QC_CLASS].astype(str) == "OK"
            if gschema.QC_CLASS in accepted.columns
            else pd.Series(False, index=accepted.index)
        )
        ok = accepted[ok_mask]
        accepted_build_ids = set(ok[gschema.BUILD_ID].astype(str))
    if accepted_build_ids and not still_failed.empty and "build_id" in still_failed.columns:
        still_failed = still_failed[~still_failed["build_id"].astype(str).isin(accepted_build_ids)]
    return (
        attempts.to_dict("records"),
        accepted.to_dict("records"),
        still_failed.to_dict("records"),
    )


def _reports_for_current_regeneration_queue(
    queue: pd.DataFrame,
    attempts: pd.DataFrame,
    accepted: pd.DataFrame,
    still_failed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop stale report rows and remap retained rows to the current queue order."""
    current = queue.reset_index(drop=True).copy()
    current["_current_queue_index"] = range(1, len(current) + 1)
    current_ids = set(current[gschema.BUILD_ID].astype(str))
    index_by_id = dict(zip(
        current[gschema.BUILD_ID].astype(str),
        current["_current_queue_index"].astype(int),
    ))

    def normalize(frame: pd.DataFrame, dedup_columns: list[str]) -> pd.DataFrame:
        if frame.empty or gschema.BUILD_ID not in frame.columns:
            return frame
        result = frame[frame[gschema.BUILD_ID].astype(str).isin(current_ids)].copy()
        if result.empty:
            return result
        result[gschema.BUILD_ID] = result[gschema.BUILD_ID].astype(str)
        result["_current_queue_index"] = result[gschema.BUILD_ID].map(index_by_id)
        old_indexes = pd.to_numeric(result.get("queue_index"), errors="coerce")
        result["_matches_current_queue"] = old_indexes.eq(result["_current_queue_index"])
        # A matching queue index identifies rows from the current queue layout.
        # Stable ordering keeps the latest append when a shard was resumed.
        result = result.sort_values("_matches_current_queue", kind="stable")
        result = result.drop_duplicates(dedup_columns, keep="last")
        result["queue_index"] = result["_current_queue_index"].astype(int)
        return result.drop(columns=["_current_queue_index", "_matches_current_queue"])

    attempts = normalize(attempts, [gschema.BUILD_ID, "attempt"])
    accepted = normalize(accepted, [gschema.BUILD_ID])
    still_failed = normalize(still_failed, [gschema.BUILD_ID])

    if not accepted.empty:
        ok_mask = (
            accepted[gschema.QC_CLASS].astype(str).eq("OK")
            if gschema.QC_CLASS in accepted.columns
            else pd.Series(False, index=accepted.index)
        )
        if "accepted_for_clean_3d_features" in accepted.columns:
            ok_mask &= accepted["accepted_for_clean_3d_features"].map(_truthy_csv_value)
        accepted = accepted[ok_mask].copy()
    if not accepted.empty and not still_failed.empty:
        accepted_ids = set(accepted[gschema.BUILD_ID].astype(str))
        still_failed = still_failed[
            ~still_failed[gschema.BUILD_ID].astype(str).isin(accepted_ids)
        ].copy()
    return attempts, accepted, still_failed


def _dual_qc_accepted_for_queue(
    accepted: pd.DataFrame,
    queue: pd.DataFrame,
    args,
) -> pd.DataFrame:
    """Revalidate every reported acceptance against its current queue spec."""
    if accepted.empty or "build_id" not in accepted.columns:
        return accepted
    specs = _read_csv_if_exists(Path(args.specs))
    specs_by_id = _spec_lookup(specs) if not specs.empty else {}
    queue_by_id = {
        str(row["build_id"]): row for _, row in queue.iterrows()
        if _safe_text(row.get("build_id", ""))
    }
    valid_rows: list[dict] = []
    for report_row in accepted.to_dict("records"):
        build_id = _safe_text(report_row.get("build_id", ""))
        queue_row = queue_by_id.get(build_id)
        if queue_row is None:
            continue
        spec, note = _regeneration_spec(queue_row, specs_by_id)
        if spec is None:
            print(f"Dropping accepted {build_id}: invalid current queue spec ({note})")
            continue
        xyz_path = Path(_safe_text(report_row.get(gschema.ACCEPTED_XYZ_PATH, "")))
        if not valid_xyz(xyz_path):
            print(f"Dropping accepted {build_id}: accepted XYZ missing or invalid: {xyz_path}")
            continue
        file_status, file_note = qc_xyz(xyz_path, spec)
        nearest = nearest_corecn_qc_xyz(
            xyz_path,
            spec,
            long_bond_threshold=float(args.long_bond_threshold),
            borderline_longish_threshold=float(args.borderline_longish_threshold),
            ambiguous_gap_threshold=float(args.ambiguous_gap_threshold),
        )
        if file_status != "accepted" or nearest.get("qc_class") != "OK":
            print(
                f"Dropping accepted {build_id}: dual QC failed "
                f"file={file_status}, nearest={nearest.get('qc_class')}"
            )
            continue
        updated = dict(report_row)
        updated.update({
            "file_qc_status": file_status,
            "file_qc_note": file_note,
            "qc_class": "OK",
            "coreCN_max_dist": nearest.get("coreCN_max_dist", ""),
            "gap_after_coreCN": nearest.get("gap_after_coreCN", ""),
            "nearest_coreCN_sig": nearest.get("nearest_coreCN_sig", ""),
            "accepted_for_clean_3d_features": True,
        })
        valid_rows.append(updated)
    return pd.DataFrame(valid_rows, columns=accepted.columns) if valid_rows else accepted.iloc[0:0].copy()


def plan_regeneration_shards(args) -> int:
    if not Path(args.regenerate_input).exists():
        print(f"Missing regeneration input: {args.regenerate_input}")
        return 1
    if int(args.num_shards) < 1:
        raise SystemExit("--num-shards must be >= 1")

    queue = pd.read_csv(args.regenerate_input, low_memory=False)
    input_rows = len(queue)
    queue = _regeneration_queue_rows(queue)
    queue = queue.copy()
    queue["queue_index"] = range(1, len(queue) + 1)
    queue["shard_id"] = (queue["queue_index"] - 1) % int(args.num_shards)

    errors: list[str] = []
    if queue["queue_index"].duplicated().any():
        errors.append("duplicate queue_index values")
    if "build_id" in queue.columns:
        dup_build_ids = queue.loc[queue["build_id"].astype(str).duplicated(keep=False), "build_id"].astype(str)
        if not dup_build_ids.empty:
            preview = ", ".join(sorted(set(dup_build_ids))[:10])
            errors.append(f"duplicate build_id values in queue: {preview}")

    expected = set(range(1, len(queue) + 1))
    assigned = set(queue["queue_index"].astype(int))
    missing = sorted(expected - assigned)
    extra = sorted(assigned - expected)
    if missing:
        errors.append(f"missing queue_index values: {missing[:10]}")
    if extra:
        errors.append(f"unexpected queue_index values: {extra[:10]}")

    counts = queue["shard_id"].value_counts().reindex(range(int(args.num_shards)), fill_value=0).sort_index()
    print("Regeneration shard dry-run")
    print(f"Input file: {args.regenerate_input}")
    print(f"Rows in input: {input_rows}")
    print(f"Regeneration rows queued: {len(queue)}")
    print(f"num_shards: {int(args.num_shards)}")
    print(f"rows_per_shard_min: {int(counts.min()) if len(counts) else 0}")
    print(f"rows_per_shard_max: {int(counts.max()) if len(counts) else 0}")
    print("Shard row counts:")
    for shard_id, count in counts.items():
        print(f"  shard {int(shard_id):02d}: {int(count)}")
    if errors:
        print("Shard invariant check: FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Shard invariant check: OK")
    return 0


def prepare_missing_regeneration(args) -> int:
    """Build a focused queue for rows whose regeneration produced no XYZ."""
    queue_path = Path(args.regenerate_input)
    still_failed_path = Path(args.still_failed_input)
    output_path = Path(args.rescue_queue_output)

    for path, label in (
        (queue_path, "regeneration input"),
        (still_failed_path, "still-failed report"),
    ):
        if not path.exists():
            print(f"Missing {label}: {path}")
            return 1

    queue = pd.read_csv(queue_path, low_memory=False)
    still_failed = pd.read_csv(still_failed_path, low_memory=False)
    required_queue = {"build_id", "qc_class"}
    required_failed = {"build_id", "best_xyz_path", "failure_note"}
    missing_queue = sorted(required_queue - set(queue.columns))
    missing_failed = sorted(required_failed - set(still_failed.columns))
    if missing_queue:
        raise SystemExit(f"{queue_path} missing columns: {missing_queue}")
    if missing_failed:
        raise SystemExit(f"{still_failed_path} missing columns: {missing_failed}")

    queue = queue[queue["qc_class"].astype(str) == "FAIL_LONG_BOND"].copy()
    if queue["build_id"].astype(str).duplicated().any():
        raise SystemExit(f"{queue_path} contains duplicate build_id values")

    failed = still_failed.copy()
    no_xyz = failed["best_xyz_path"].fillna("").astype(str).str.strip().eq("")
    if "accepted_for_clean_3d_features" in failed.columns:
        accepted = failed["accepted_for_clean_3d_features"].map(_truthy_csv_value)
        no_xyz &= ~accepted
    failed = failed.loc[no_xyz].copy()
    if failed["build_id"].astype(str).duplicated().any():
        raise SystemExit(f"{still_failed_path} contains duplicate no-XYZ build_id values")

    failure_by_id = {
        str(row["build_id"]): _safe_text(row.get("failure_note", ""))
        for row in failed.to_dict("records")
    }
    rescue_ids = set(failure_by_id)
    rescue = queue[queue["build_id"].astype(str).isin(rescue_ids)].copy()
    matched_ids = set(rescue["build_id"].astype(str))
    missing_ids = sorted(rescue_ids - matched_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise SystemExit(
            f"{len(missing_ids)} no-XYZ build_id values are absent from {queue_path}: {preview}"
        )

    rescue["rescue_previous_failure"] = rescue["build_id"].astype(str).map(failure_by_id)
    rescue["rescue_reason"] = "regeneration_produced_no_xyz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rescue.to_csv(output_path, index=False)

    failure_counts = rescue["rescue_previous_failure"].value_counts(dropna=False)
    print("Missing-geometry regeneration rescue queue")
    print(f"Original FAIL_LONG_BOND rows: {len(queue)}")
    print(f"Still-failed rows: {len(still_failed)}")
    print(f"Rows with no generated XYZ: {len(rescue)}")
    print(f"Output: {output_path}")
    print("Previous failure counts:")
    for failure, count in failure_counts.items():
        print(f"  {failure or '<blank>'}: {int(count)}")
    return 0


def _write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _selected_azole_is_ambiguous(smiles: str, coord_list: list[int]) -> bool:
    """Flag five-membered azoles with >1 selected N from the same ring."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        mol = None
    if mol is None:
        return True
    selected = set(coord_list)
    return any(
        len(ring) == 5
        and sum(
            idx in selected and mol.GetAtomWithIdx(idx).GetSymbol() == "N"
            for idx in ring
        ) > 1
        for ring in mol.GetRingInfo().AtomRings()
    )


def _family_for_replanned_donors(smiles: str, donors) -> tuple[str, str]:
    """Return a chemistry family label and an operational run group."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        mol = None
    atom_symbols = [] if mol is None else [atom.GetSymbol() for atom in mol.GetAtoms()]
    carboxyl_count = 0
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue
            bonded_oxygen_types = [
                mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType()
                for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "O"
            ]
            if (Chem.BondType.DOUBLE in bonded_oxygen_types
                    and Chem.BondType.SINGLE in bonded_oxygen_types):
                carboxyl_count += 1

    denticity = donors.denticity
    if donors.strategy == "polypyridyl_triazine":
        names = {3: "BTP_like_N3", 4: "BTBP_BTPhen_like_N4", 5: "BTTP_like_N5"}
        return names.get(denticity, f"polypyridyl_triazine_N{denticity}"), "template_pincer"
    if donors.strategy == "pytri":
        return f"PyTri_like_N{denticity}", "template_pincer"
    if donors.strategy == "flavonol_3hydroxy_4oxo":
        return "flavonol_3OH_4oxo_bidentate", "template_flavonol"
    if carboxyl_count >= 3 and donors.donor_types.count("N(amine)") >= 2:
        return f"aminopolycarboxylate_{denticity}dent", "template_aminopolycarboxylate"
    if donors.strategy == "compact_amide_core":
        return f"compact_amide_core_{denticity}dent", "template_amide_core"
    if atom_symbols.count("P"):
        return f"phosphoryl_thiophosphoryl_{denticity}dent", "template_other"
    if mol is not None and mol.GetNumHeavyAtoms() >= HEAVY_LIGAND_ATOM_THRESHOLD:
        return f"large_rigid_filtered_{denticity}dent", "template_large_rigid"
    return f"filtered_donor_pocket_{denticity}dent", "template_other"


def _suggest_ligtype(donors) -> str:
    """Family-safe initial Architector placement for a corrected donor set."""
    denticity = donors.denticity
    if denticity == 1:
        return "mono"
    if denticity == 2:
        return "bi_cis_chelating"
    if denticity == 3:
        return "tri_mer_bent"
    if denticity == 4:
        if donors.strategy == "compact_amide_core" and "N(amine)" in donors.donor_types:
            return "tetra_trigonal_pyramidal"
        return "tetra_planar_bent"
    if denticity == 5:
        return "penta_planar_bent"
    if denticity == 6:
        return "hexa_octahedral"
    return ""


def _frame_or_empty(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=columns)


def _family_template_values(row, smiles: str, donors) -> dict:
    """Return old and proposed full coordination-template values.

    A donor-list correction alone is not sufficient for BTP/BTBP or very large
    chelators: the family may also impose a different core CN, ligand
    stoichiometry and fill count.  Keeping the comparison in one helper makes
    the targeted rescue and full stage-1 migration audit use identical rules.
    """
    old_coord = _coordlist(row.get("COORDLIST", ""))
    old_dent = _safe_int(row.get("DENTATE"), len(old_coord))
    old_core_cn = _safe_int(row.get("coreCN"), 0)
    old_n_ligs = _safe_int(row.get("n_ligs"), 0)
    old_n_fill = _safe_int(row.get("n_fill"), 0)

    proposed_core_cn = core_cn_for_donor_set(old_core_cn, donors)
    proposed_n_ligs = choose_n_ligs_for_donor_set(
        proposed_core_cn, donors, ligand_smiles=smiles,
    )
    proposed_n_fill = coordination_n_fill(
        proposed_core_cn, donors.denticity, proposed_n_ligs,
    )
    template_changed = any((
        old_coord != donors.coord_list,
        old_dent != donors.denticity,
        old_core_cn != proposed_core_cn,
        old_n_ligs != proposed_n_ligs,
        old_n_fill != proposed_n_fill,
    ))
    return {
        "old_coord": old_coord,
        "old_dent": old_dent,
        "old_core_cn": old_core_cn,
        "old_n_ligs": old_n_ligs,
        "old_n_fill": old_n_fill,
        "proposed_core_cn": proposed_core_cn,
        "proposed_n_ligs": proposed_n_ligs,
        "proposed_n_fill": proposed_n_fill,
        "template_changed": template_changed,
    }


def prepare_family_regeneration(args) -> int:
    """Split current failures into chemistry-specific, non-overlapping queues.

    Corrected donor templates receive a *new* build_id, so the old geometry and
    its frozen metadata remain untouched.  The new id is identical to the id a
    fresh stage-1 run will produce from the corrected coordination planner.
    """
    queue_path = Path(args.regenerate_input)
    still_failed_path = Path(args.still_failed_input)
    output_dir = Path(args.family_plan_dir)
    accepted_path = Path(args.reports_dir) / REGENERATE_ACCEPTED_NAME
    for path, label in (
        (queue_path, "regeneration input"),
        (still_failed_path, "still-failed report"),
    ):
        if not path.exists():
            print(f"Missing {label}: {path}")
            return 1

    queue = pd.read_csv(queue_path, low_memory=False)
    failed = pd.read_csv(still_failed_path, low_memory=False)
    accepted = pd.read_csv(accepted_path, low_memory=False) if accepted_path.exists() else pd.DataFrame()
    if "qc_class" not in queue.columns or "build_id" not in queue.columns:
        raise SystemExit(f"{queue_path} needs build_id and qc_class")
    if "build_id" not in failed.columns:
        raise SystemExit(f"{still_failed_path} needs build_id")
    queue = queue[queue["qc_class"].astype(str) == "FAIL_LONG_BOND"].copy()
    for frame, path in ((queue, queue_path), (failed, still_failed_path)):
        if frame["build_id"].astype(str).duplicated().any():
            raise SystemExit(f"{path} contains duplicate build_id values")

    accepted_ids = (
        set(accepted["build_id"].astype(str)) if "build_id" in accepted.columns else set()
    )
    failed = failed[~failed["build_id"].astype(str).isin(accepted_ids)].copy()
    failed_ids = set(failed["build_id"].astype(str))
    current = queue[queue["build_id"].astype(str).isin(failed_ids)].copy()
    missing_ids = sorted(failed_ids - set(current["build_id"].astype(str)))
    if missing_ids:
        raise SystemExit(
            f"{len(missing_ids)} still-failed ids are absent from {queue_path}: "
            + ", ".join(missing_ids[:10])
        )

    detail_columns = [
        name for name in (
            "best_qc_class", "best_coreCN_max_dist", "best_gap_after_coreCN",
            "best_xyz_path", "best_failed_xyz_path", "failure_note",
        ) if name in failed.columns
    ]
    details = failed.set_index(failed["build_id"].astype(str))
    for name in detail_columns:
        current[name] = current["build_id"].astype(str).map(details[name])

    accepted_smiles = (
        set(accepted["smiles_for_architector_used"].dropna().astype(str))
        if "smiles_for_architector_used" in accepted.columns else set()
    )
    routes: dict[str, list[dict]] = {
        "template_replan": [],
        "conformer_borderline": [],
        "conformer_sibling": [],
        "conformer_deep": [],
        "no_xyz_uff": [],
        "no_xyz_ligtype": [],
        "manual_review": [],
    }
    ligtype_rows: list[dict] = []

    for _, row in current.iterrows():
        source_build_id = str(row["build_id"])
        smiles = _safe_text(row.get("smiles_for_architector_used", "")) or _safe_text(
            row.get("SMILES_FOR_ARCHITECTOR", "")
        )
        donors = detect_donors(smiles)
        base = row.to_dict()
        base["source_build_id"] = source_build_id
        if donors is None:
            base.update({
                "rescue_route": "manual_review",
                "ligand_family": "unparseable_or_no_donors",
                "review_reason": "corrected_donor_detection_failed",
            })
            routes["manual_review"].append(base)
            continue

        values = _family_template_values(row, smiles, donors)
        old_coord = values["old_coord"]
        ambiguous_azole = _selected_azole_is_ambiguous(smiles, donors.coord_list)
        if ambiguous_azole:
            base.update({
                "rescue_route": "manual_review",
                "ligand_family": "azole_multiple_possible_ring_n",
                "review_reason": "more_than_one_selected_N_in_same_five_membered_azole",
                "proposed_COORDLIST": json.dumps(donors.coord_list),
                "proposed_DENTATE": donors.denticity,
                "proposed_coreCN": values["proposed_core_cn"],
                "proposed_n_ligs": values["proposed_n_ligs"],
                "proposed_n_fill": values["proposed_n_fill"],
            })
            routes["manual_review"].append(base)
            continue

        if values["template_changed"]:
            family, run_group = _family_for_replanned_donors(smiles, donors)
            core_cn = values["proposed_core_cn"]
            ligand_count = values["proposed_n_ligs"]
            fill_count = values["proposed_n_fill"]
            anion = _safe_text(row.get("inner_sphere_anion", "water"), "water")
            fill_ligand = _safe_text(row.get("fill_ligand", anion), anion)
            metal_z = int(float(row["Atomic Number_metal"]))
            new_build_id = complex_build_id(
                metal_Z=metal_z,
                ligand_smiles=smiles,
                coord_list=donors.coord_list,
                denticity=donors.denticity,
                core_cn=core_cn,
                n_ligs=ligand_count,
                inner_sphere_anion=anion,
                fill_ligand=fill_ligand,
                n_fill_value=fill_count,
            )
            proposed = dict(base)
            proposed.update({
                "build_id": new_build_id,
                "SMILES_FOR_ARCHITECTOR": smiles,
                "smiles_for_architector_used": smiles,
                "COORDLIST": json.dumps(donors.coord_list),
                "DONOR_TYPES": json.dumps(donors.donor_types),
                "DENTATE": donors.denticity,
                "coreCN": core_cn,
                "n_ligs": ligand_count,
                "n_fill": fill_count,
                "rescue_route": "template_replan",
                "run_group": run_group,
                "ligand_family": family,
                "donor_strategy": donors.strategy,
                "original_COORDLIST": json.dumps(old_coord),
                "original_DENTATE": values["old_dent"],
                "original_coreCN": values["old_core_cn"],
                "original_n_ligs": values["old_n_ligs"],
                "original_n_fill": values["old_n_fill"],
                "proposed_COORDLIST": json.dumps(donors.coord_list),
                "proposed_DENTATE": donors.denticity,
                "proposed_coreCN": core_cn,
                "proposed_n_ligs": ligand_count,
                "proposed_n_fill": fill_count,
                "template_change_note": (
                    f"donors {len(old_coord)}->{donors.denticity}; "
                    f"coreCN {values['old_core_cn']}->{core_cn}; "
                    f"n_ligs {values['old_n_ligs']}->{ligand_count}; "
                    f"n_fill {values['old_n_fill']}->{fill_count}; "
                    "new build_id preserves original geometry"
                ),
            })
            routes["template_replan"].append(proposed)
            ligtype = _suggest_ligtype(donors)
            if ligtype:
                ligtype_rows.append({
                    "enabled": 1,
                    "build_id": new_build_id,
                    "geometry_key": _safe_text(row.get("geometry_key", "")),
                    "ligType": ligtype,
                    "DENTATE": donors.denticity,
                    "COORDLIST": json.dumps(donors.coord_list),
                    "DONOR_TYPES": json.dumps(donors.donor_types),
                    "SMILES_FOR_ARCHITECTOR": smiles,
                    "metal_symbol": _safe_text(row.get("metal_symbol", "")),
                    "coreCN": core_cn,
                    "n_ligs": ligand_count,
                    "inner_sphere_anion": anion,
                    "note": f"family template replan: {family}",
                })
            continue

        old_dent = values["old_dent"]
        if old_dent >= 6:
            base.update({
                "rescue_route": "manual_review",
                "ligand_family": "unresolved_high_denticity",
                "review_reason": "high_denticity_unchanged_by_safe_family_rules",
            })
            routes["manual_review"].append(base)
            continue

        best_xyz = _safe_text(row.get("best_xyz_path", ""))
        if not best_xyz:
            failure_note = _safe_text(row.get("failure_note", "")).lower()
            route = "no_xyz_ligtype" if "cannot assign lig" in failure_note else "no_xyz_uff"
            base.update({
                "rescue_route": route,
                "ligand_family": "valid_template_no_candidate",
                "review_reason": failure_note,
            })
            routes[route].append(base)
            continue

        best_dist = _safe_float(row.get("best_coreCN_max_dist", math.nan))
        if math.isfinite(best_dist) and best_dist <= 3.30:
            route = "conformer_borderline"
        elif smiles in accepted_smiles:
            route = "conformer_sibling"
        else:
            route = "conformer_deep"
        base.update({
            "rescue_route": route,
            "ligand_family": f"valid_template_{donors.denticity}dent",
            "review_reason": "fixed_template_conformer_search",
        })
        routes[route].append(base)

    # A stage-1 rerun will apply the corrected detector to every frozen spec,
    # not only today's failures.  Audit that full migration surface, but keep
    # non-failing rows out of the automatic rescue until the family pilots have
    # passed geometry QC.
    route_by_source = {
        str(row["source_build_id"]): route
        for route, rows in routes.items()
        for row in rows
    }
    all_template_changes: list[dict] = []
    specs_path = Path(getattr(args, "specs", DEFAULT_SPECS))
    if specs_path.exists():
        frozen_specs = pd.read_csv(specs_path, low_memory=False)
        for _, spec in frozen_specs.iterrows():
            smiles = _safe_text(spec.get("SMILES_FOR_ARCHITECTOR", ""))
            donors = detect_donors(smiles)
            if donors is None:
                continue
            values = _family_template_values(spec, smiles, donors)
            if not values["template_changed"]:
                continue
            old_coord = values["old_coord"]
            source_build_id = str(spec["build_id"])
            core_cn = values["proposed_core_cn"]
            ligand_count = values["proposed_n_ligs"]
            fill_count = values["proposed_n_fill"]
            anion = _safe_text(spec.get("inner_sphere_anion", "water"), "water")
            fill_ligand = _safe_text(spec.get("fill_ligand", anion), anion)
            new_build_id = complex_build_id(
                metal_Z=int(float(spec["Atomic Number_metal"])),
                ligand_smiles=smiles,
                coord_list=donors.coord_list,
                denticity=donors.denticity,
                core_cn=core_cn,
                n_ligs=ligand_count,
                inner_sphere_anion=anion,
                fill_ligand=fill_ligand,
                n_fill_value=fill_count,
            )
            family, run_group = _family_for_replanned_donors(smiles, donors)
            ambiguous = _selected_azole_is_ambiguous(smiles, donors.coord_list)
            all_template_changes.append({
                "source_build_id": source_build_id,
                "proposed_build_id": new_build_id,
                "geometry_key": _safe_text(spec.get("geometry_key", "")),
                "metal_symbol": _safe_text(spec.get("metal_symbol", "")),
                "inner_sphere_anion": anion,
                "SMILES_FOR_ARCHITECTOR": smiles,
                "original_COORDLIST": json.dumps(old_coord),
                "proposed_COORDLIST": json.dumps(donors.coord_list),
                "original_DENTATE": values["old_dent"],
                "proposed_DENTATE": donors.denticity,
                "original_coreCN": values["old_core_cn"],
                "proposed_coreCN": core_cn,
                "original_n_ligs": values["old_n_ligs"],
                "proposed_n_ligs": ligand_count,
                "original_n_fill": values["old_n_fill"],
                "proposed_n_fill": fill_count,
                "donor_strategy": donors.strategy,
                "ligand_family": family,
                "run_group": run_group,
                "suggested_ligType": _suggest_ligtype(donors),
                "migration_scope": (
                    "current_still_failed" if source_build_id in failed_ids
                    else "deferred_nonfailed_audit"
                ),
                "current_rescue_route": route_by_source.get(source_build_id, ""),
                "manual_review_required": bool(ambiguous),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    base_columns = list(current.columns) + [
        "source_build_id", "rescue_route", "run_group", "ligand_family",
        "donor_strategy", "review_reason", "original_COORDLIST", "original_DENTATE",
        "original_coreCN", "original_n_ligs", "original_n_fill",
        "proposed_COORDLIST", "proposed_DENTATE", "proposed_coreCN",
        "proposed_n_ligs", "proposed_n_fill", "template_change_note",
    ]
    frames = {
        route: _frame_or_empty(rows, base_columns)
        for route, rows in routes.items()
    }
    for route, frame in frames.items():
        _write_csv_atomic(frame, output_dir / f"{route}.csv")

    replan = frames["template_replan"]
    run_groups = [
        "template_pincer", "template_aminopolycarboxylate", "template_amide_core",
        "template_large_rigid", "template_flavonol", "template_other",
    ]
    for group in run_groups:
        part = (
            replan[replan["run_group"].astype(str) == group].copy()
            if not replan.empty else replan.copy()
        )
        _write_csv_atomic(part, output_dir / f"{group}.csv")

    ligtype_frame = pd.DataFrame(ligtype_rows, columns=[
        "enabled", "build_id", "geometry_key", "ligType", "DENTATE", "COORDLIST",
        "DONOR_TYPES", "SMILES_FOR_ARCHITECTOR", "metal_symbol", "coreCN", "n_ligs",
        "inner_sphere_anion", "note",
    ])
    _write_csv_atomic(ligtype_frame, output_dir / "template_ligtype_overrides.csv")

    config = {
        "template_pincer": ("standard:large_ligand_fast", 2, 3600, 16, 4, "3G"),
        "template_aminopolycarboxylate": ("standard:large_ligand_deep", 2, 3600, 8, 3, "3G"),
        "template_amide_core": ("standard:large_ligand_fast", 2, 3000, 6, 3, "3G"),
        "template_large_rigid": ("large_ligand_fast:large_ligand_deep", 2, 5400, 12, 3, "5G"),
        "template_flavonol": ("large_ligand_fast:standard", 2, 3600, 2, 1, "3G"),
        "template_other": ("standard:large_ligand_fast", 2, 3600, 6, 3, "3G"),
        "conformer_borderline": ("large_ligand_fast:standard", 2, 2400, 8, 4, "3G"),
        "conformer_sibling": ("standard:large_ligand_fast", 2, 3000, 4, 2, "3G"),
        "conformer_deep": ("large_ligand_deep:standard", 2, 5400, 6, 2, "5G"),
        "no_xyz_uff": ("uff_xtb_no_preopt:uff_unrelaxed", 2, 7200, 6, 3, "5G"),
        "no_xyz_ligtype": ("large_ligand_fast:standard", 2, 3600, 2, 1, "4G"),
    }
    config_rows = []
    for group, values in config.items():
        frame = (
            frames[group] if group in frames
            else replan[replan["run_group"].astype(str) == group]
        )
        profiles, attempts, timeout, shard_cap, concurrency, memory = values
        rows = len(frame)
        queue_path = output_dir / f"{group}.csv"
        try:
            portable_queue_path = str(queue_path.resolve().relative_to(_REPO_ROOT))
        except ValueError:
            portable_queue_path = str(queue_path)
        config_rows.append({
            "run_group": group,
            "queue": portable_queue_path,
            "rows": rows,
            "profile_sequence": profiles,
            "max_attempts": attempts,
            "timeout_per_complex": timeout,
            "num_shards": min(shard_cap, rows) if rows else 0,
            "max_concurrent_tasks": min(concurrency, rows) if rows else 0,
            "memory": memory,
        })
    config_frame = pd.DataFrame(config_rows)
    _write_csv_atomic(config_frame, output_dir / "run_config.csv")

    summary_rows = []
    for route, frame in frames.items():
        if frame.empty:
            summary_rows.append({"rescue_route": route, "ligand_family": "", "rows": 0})
            continue
        counts = frame["ligand_family"].fillna("").astype(str).value_counts()
        summary_rows.extend(
            {"rescue_route": route, "ligand_family": family, "rows": int(count)}
            for family, count in counts.items()
        )
    _write_csv_atomic(pd.DataFrame(summary_rows), output_dir / "family_summary.csv")
    all_changes_frame = pd.DataFrame(all_template_changes)
    _write_csv_atomic(all_changes_frame, output_dir / "all_template_changes_audit.csv")

    print("Family-aware regeneration plan")
    print(f"Current still-failed rows: {len(current)}")
    for route, frame in frames.items():
        print(f"  {route}: {len(frame)} -> {output_dir / f'{route}.csv'}")
    print(f"  template ligType overrides: {len(ligtype_frame)}")
    if not all_changes_frame.empty:
        deferred = int(
            all_changes_frame["migration_scope"].astype(str).eq("deferred_nonfailed_audit").sum()
        )
        print(
            f"  all frozen template changes: {len(all_changes_frame)} "
            f"({deferred} deferred non-failing specs)"
        )
    print(f"  run config: {output_dir / 'run_config.csv'}")
    return 0


def prepare_adaptive_regeneration(args) -> int:
    """Create one fill-aware adjacent-CN hypothesis for diagnosed failures.

    ``nitrate_bi`` occupies two coordination sites and Architector uses water
    as its secondary fill ligand.  An odd number of open sites therefore
    forces one extra water into the shell.  When two builds already produced a
    compact but ambiguous shell, propose the adjacent CN8/CN9 value only if it
    removes that odd site.  Donors and ligand stoichiometry remain fixed, and a
    new build ID preserves the original hypothesis and geometry.
    """
    input_path = Path(args.adaptive_input)
    output_path = Path(args.adaptive_output)
    if not input_path.exists():
        print(f"Missing adaptive regeneration input: {input_path}")
        return 1

    failed = pd.read_csv(input_path, low_memory=False)
    required = {
        "build_id", "best_qc_class", "attempts_run", "coreCN", "DENTATE", "n_ligs",
        "COORDLIST", "Atomic Number_metal", "inner_sphere_anion",
    }
    missing = sorted(required - set(failed.columns))
    if missing:
        raise SystemExit(f"{input_path} is missing required columns: {missing}")

    attempts_path = getattr(args, "adaptive_attempts_input", None)
    if attempts_path is None:
        attempts_path = input_path.with_name(REGENERATE_ATTEMPTS_NAME)
    attempts = _deduplicate_regeneration_attempts(
        _read_csv_if_exists(Path(attempts_path))
    )
    ambiguous_attempt_counts: dict[str, int] = {}
    if not attempts.empty and {"build_id", "qc_class"}.issubset(attempts.columns):
        ambiguous = attempts[attempts["qc_class"].astype(str).eq("BORDERLINE_AMBIGUOUS_SHELL")]
        ambiguous_attempt_counts = {
            str(build_id): int(count)
            for build_id, count in ambiguous.groupby(ambiguous["build_id"].astype(str)).size().items()
        }

    proposals: list[dict] = []
    excluded: dict[str, int] = {}

    def reject(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for row in failed.to_dict("records"):
        if _safe_text(row.get("best_qc_class")) != "BORDERLINE_AMBIGUOUS_SHELL":
            reject("not_ambiguous_shell")
            continue
        # Aggregate counters in a still-failed report can be stale after a
        # changed queue layout.  Require two concrete attempt records carrying
        # this diagnosis before changing the coordination hypothesis.
        diagnosed_attempts = ambiguous_attempt_counts.get(
            _safe_text(row.get("build_id")), 0,
        )
        if diagnosed_attempts < 2:
            reject("diagnosis_not_repeated")
            continue
        best_distance = _safe_float(row.get("best_coreCN_max_dist"))
        if not math.isfinite(best_distance) or best_distance > float(args.long_bond_threshold):
            reject("long_or_missing_core_distance")
            continue

        anion = _safe_text(row.get("inner_sphere_anion"), "water").lower()
        fill_ligand = _safe_text(row.get("fill_ligand"), anion).lower()
        if anion != "nitrate" or fill_ligand not in {"nitrate", "nitrate_bi"}:
            reject("not_bidentate_nitrate_fill")
            continue

        core_cn = _safe_int(row.get("coreCN"), 0)
        denticity = _safe_int(row.get("DENTATE"), 0)
        n_ligs = _safe_int(row.get("n_ligs"), 0)
        donor_sites = denticity * n_ligs
        open_sites = core_cn - donor_sites
        if core_cn not in {8, 9} or denticity < 1 or n_ligs < 1 or open_sites < 1:
            reject("invalid_or_nonadjacent_cn_template")
            continue

        current_secondary_water = open_sites % 2
        candidates = []
        for candidate_cn in (8, 9):
            if candidate_cn == core_cn:
                continue
            candidate_open_sites = candidate_cn - donor_sites
            if candidate_open_sites < 0:
                continue
            candidate_secondary_water = candidate_open_sites % 2
            if candidate_secondary_water < current_secondary_water:
                candidates.append((candidate_secondary_water, candidate_cn, candidate_open_sites))
        if not candidates:
            reject("adjacent_cn_does_not_improve_fill_parity")
            continue

        _, proposed_cn, proposed_open_sites = min(candidates)
        smiles = _safe_text(row.get("smiles_for_architector_used")) or _safe_text(
            row.get("SMILES_FOR_ARCHITECTOR")
        )
        coord_list = _coordlist(row.get("COORDLIST", ""))
        if not smiles or not coord_list:
            reject("missing_smiles_or_coordlist")
            continue

        metal_z = _safe_int(row.get("Atomic Number_metal"), 0)
        if metal_z < 1:
            reject("missing_metal_atomic_number")
            continue
        source_build_id = _safe_text(row.get("build_id"))
        new_build_id = complex_build_id(
            metal_Z=metal_z,
            ligand_smiles=smiles,
            coord_list=coord_list,
            denticity=denticity,
            core_cn=proposed_cn,
            n_ligs=n_ligs,
            inner_sphere_anion=anion,
            fill_ligand="nitrate" if fill_ligand == "nitrate_bi" else fill_ligand,
            n_fill_value=proposed_open_sites,
        )
        proposal = dict(row)
        proposal.update({
            "source_build_id": source_build_id,
            "build_id": new_build_id,
            "SMILES_FOR_ARCHITECTOR": smiles,
            "smiles_for_architector_used": smiles,
            "coreCN": proposed_cn,
            "n_fill": proposed_open_sites,
            "fill_ligand": "nitrate",
            "source_xyz_path": _safe_text(row.get("best_xyz_path")),
            "rescue_route": "adaptive_cn_fill",
            "hypothesis_version": ADAPTIVE_CN_FILL_VERSION,
            "parent_qc_diagnosis": "repeated_ambiguous_shell",
            "original_coreCN": core_cn,
            "original_n_fill": open_sites,
            "original_nitrate_count": open_sites // 2,
            "original_secondary_water_count": current_secondary_water,
            "proposed_coreCN": proposed_cn,
            "proposed_n_fill": proposed_open_sites,
            "proposed_nitrate_count": proposed_open_sites // 2,
            "proposed_secondary_water_count": proposed_open_sites % 2,
            "adaptive_reason": (
                "adjacent_CN_removes_odd_open_site_for_bidentate_nitrate_fill"
            ),
        })
        proposals.append(proposal)

    output = pd.DataFrame(proposals)
    if output.empty:
        output_columns = [
            *failed.columns, "source_build_id", "rescue_route", "hypothesis_version",
            "parent_qc_diagnosis", "original_coreCN", "original_n_fill",
            "original_nitrate_count", "original_secondary_water_count",
            "proposed_coreCN", "proposed_n_fill", "proposed_nitrate_count",
            "proposed_secondary_water_count", "adaptive_reason", "source_xyz_path",
        ]
        output = pd.DataFrame(columns=list(dict.fromkeys(output_columns)))
    _write_csv_atomic(output, output_path)

    print("Adaptive CN/fill regeneration queue")
    print(f"Input failures: {len(failed)}")
    print(f"Proposed adjacent-CN hypotheses: {len(output)} -> {output_path}")
    for reason, count in sorted(excluded.items()):
        print(f"  excluded {reason}: {count}")
    return 0


def family_regeneration_status(args) -> int:
    """Summarise accepted, failed and pending rows for every family queue."""
    plan_dir = Path(args.family_plan_dir)
    runs_dir = Path(args.family_runs_dir)
    config_path = plan_dir / "run_config.csv"
    if not config_path.exists():
        print(f"Missing family run config: {config_path}")
        print("Run prepare-family-regeneration first.")
        return 1

    config = pd.read_csv(config_path, low_memory=False)
    status_rows: list[dict] = []
    for _, config_row in config.iterrows():
        group = str(config_row["run_group"])
        planned = _safe_int(config_row.get("rows"), 0)
        report_dir = runs_dir / group
        accepted_frames = [
            _read_csv_if_exists(report_dir / REGENERATE_ACCEPTED_NAME),
            _read_all_regen_shard_reports(report_dir, REGENERATE_ACCEPTED_NAME),
        ]
        failed_frames = [
            _read_csv_if_exists(report_dir / REGENERATE_STILL_FAILED_NAME),
            _read_all_regen_shard_reports(report_dir, REGENERATE_STILL_FAILED_NAME),
        ]
        accepted = pd.concat(
            [frame for frame in accepted_frames if not frame.empty],
            ignore_index=True, sort=False,
        ) if any(not frame.empty for frame in accepted_frames) else pd.DataFrame()
        failed = pd.concat(
            [frame for frame in failed_frames if not frame.empty],
            ignore_index=True, sort=False,
        ) if any(not frame.empty for frame in failed_frames) else pd.DataFrame()

        # A report directory can contain earlier pilot cardinalities.  Keep
        # durable outcomes only for the currently planned build IDs.
        queue_path = plan_dir / f"{group}.csv"
        planned_ids: set[str] | None = None
        if queue_path.exists():
            planned_queue = _read_csv_if_exists(queue_path)
            if "build_id" in planned_queue.columns:
                planned_ids = set(planned_queue["build_id"].astype(str))
                planned = len(planned_ids)
        if planned_ids is not None:
            if not accepted.empty and "build_id" in accepted.columns:
                accepted = accepted[accepted["build_id"].astype(str).isin(planned_ids)]
            if not failed.empty and "build_id" in failed.columns:
                failed = failed[failed["build_id"].astype(str).isin(planned_ids)]

        accepted_ids: set[str] = set()
        if not accepted.empty and "build_id" in accepted.columns:
            ok_mask = (
                accepted["qc_class"].astype(str).eq("OK")
                if "qc_class" in accepted.columns
                else pd.Series(False, index=accepted.index)
            )
            if "accepted_for_clean_3d_features" in accepted.columns:
                ok_mask &= accepted["accepted_for_clean_3d_features"].map(_truthy_csv_value)
            accepted_ids = set(accepted.loc[ok_mask, "build_id"].astype(str))

        failed_ids = (
            set(failed["build_id"].astype(str)) - accepted_ids
            if not failed.empty and "build_id" in failed.columns else set()
        )
        completed = len(accepted_ids | failed_ids)
        pending = max(0, planned - completed)
        if planned == 0:
            state = "empty"
        elif pending == 0:
            state = "complete"
        elif completed:
            state = "partial"
        else:
            state = "pending"
        status_rows.append({
            "run_group": group,
            "planned": planned,
            "accepted": len(accepted_ids),
            "still_failed": len(failed_ids),
            "pending": pending,
            "state": state,
            "reports_dir": str(report_dir),
        })

    status = pd.DataFrame(status_rows)
    output_path = plan_dir / "status.csv"
    _write_csv_atomic(status, output_path)
    display_columns = ["run_group", "planned", "accepted", "still_failed", "pending", "state"]
    print("Family regeneration status")
    print(status[display_columns].to_string(index=False))
    print(f"Status CSV: {output_path}")
    return 0


def _regeneration_report_union(report_dir: Path, name: str) -> pd.DataFrame:
    frames = [
        _read_csv_if_exists(report_dir / name),
        _read_all_regen_shard_reports(report_dir, name),
    ]
    present = [frame for frame in frames if not frame.empty]
    return pd.concat(present, ignore_index=True, sort=False) if present else pd.DataFrame()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _queue_path_from_config(value, plan_dir: Path, group: str) -> Path:
    candidate = Path(_safe_text(value)) if _safe_text(value) else plan_dir / f"{group}.csv"
    if candidate.is_absolute():
        return candidate
    repo_candidate = _REPO_ROOT / candidate
    return repo_candidate if repo_candidate.exists() else plan_dir / candidate.name


def _candidate_xyz_path(value, spec: pd.Series | dict) -> Path | None:
    candidates: list[Path] = []
    text = _safe_text(value)
    if text:
        candidates.append(Path(text))
        if not Path(text).is_absolute():
            candidates.append(_REPO_ROOT / text)
    expected, _, _ = expected_paths(pd.Series(spec), GEOMETRY_DIR)
    candidates.append(expected)
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if valid_xyz(candidate):
            return candidate
    return None


def _dual_qc_candidate(value, spec: pd.Series | dict, args) -> dict:
    xyz_path = _candidate_xyz_path(value, spec)
    if xyz_path is None:
        return {
            "accepted": False,
            "candidate_exists": False,
            "xyz_path": _safe_text(value),
            "file_qc_status": "UNAVAILABLE",
            "file_qc_note": "candidate_xyz_unavailable",
            "qc_class": "UNAVAILABLE",
            "qc_note": "candidate_xyz_unavailable",
        }
    file_status, file_note = qc_xyz(xyz_path, pd.Series(spec))
    nearest = nearest_corecn_qc_xyz(
        xyz_path,
        spec,
        long_bond_threshold=float(args.long_bond_threshold),
        borderline_longish_threshold=float(args.borderline_longish_threshold),
        ambiguous_gap_threshold=float(args.ambiguous_gap_threshold),
    )
    return {
        "accepted": file_status == "accepted" and nearest.get("qc_class") == "OK",
        "candidate_exists": True,
        "xyz_path": str(xyz_path),
        "file_qc_status": file_status,
        "file_qc_note": file_note,
        **nearest,
    }


def _qc_spec_snapshot(spec: pd.Series | dict) -> str:
    fields = (
        "build_id", "SMILES_FOR_ARCHITECTOR", "metal_symbol", "Atomic Number_metal",
        "metal_ox", "COORDLIST", "DENTATE", "coreCN", "n_ligs",
        "inner_sphere_anion", "fill_ligand", "n_fill", "geometry_key",
    )
    payload = {name: _safe_text(spec.get(name, "")) for name in fields}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _new_hypothesis_row(
    row: pd.Series | dict,
    *,
    root_source_build_id: str,
    parent_build_id: str,
    route: str,
    version: str,
    core_cn: int,
    n_fill: int,
    reason: str,
) -> dict:
    proposal = dict(row)
    smiles = _safe_text(proposal.get("SMILES_FOR_ARCHITECTOR")) or _safe_text(
        proposal.get("smiles_for_architector_used")
    )
    coord = _coordlist(proposal.get("COORDLIST", ""))
    denticity = _safe_int(proposal.get("DENTATE"), len(coord))
    n_ligs = _safe_int(proposal.get("n_ligs"), 0)
    anion = _safe_text(proposal.get("inner_sphere_anion"), "water")
    fill_ligand = _safe_text(proposal.get("fill_ligand"), anion)
    build_id = complex_build_id(
        metal_Z=_safe_int(proposal.get("Atomic Number_metal"), 0),
        ligand_smiles=smiles,
        coord_list=coord,
        denticity=denticity,
        core_cn=core_cn,
        n_ligs=n_ligs,
        inner_sphere_anion=anion,
        fill_ligand=fill_ligand,
        n_fill_value=n_fill,
    )
    proposal.update({
        "root_source_build_id": root_source_build_id,
        "source_build_id": root_source_build_id,
        "parent_build_id": parent_build_id,
        "build_id": build_id,
        "SMILES_FOR_ARCHITECTOR": smiles,
        "smiles_for_architector_used": smiles,
        "coreCN": core_cn,
        "n_fill": n_fill,
        "fill_ligand": fill_ligand,
        "rescue_route": route,
        "hypothesis_version": version,
        "hypothesis_reason": reason,
    })
    return proposal


def _adaptive_cn_fill_candidate(row: dict, version: str, long_bond_threshold: float) -> tuple[dict | None, str]:
    if _safe_text(row.get("best_qc_class")) != "BORDERLINE_AMBIGUOUS_SHELL":
        return None, "not_ambiguous_shell"
    if _safe_int(row.get("ambiguous_attempts"), 0) < 2:
        return None, "diagnosis_not_repeated"
    best_distance = _safe_float(row.get("best_coreCN_max_dist"))
    if not math.isfinite(best_distance) or best_distance > float(long_bond_threshold):
        return None, "long_or_missing_core_distance"
    anion = _safe_text(row.get("inner_sphere_anion"), "water").lower()
    fill_ligand = _safe_text(row.get("fill_ligand"), anion).lower()
    if anion != "nitrate" or fill_ligand not in {"nitrate", "nitrate_bi"}:
        return None, "not_bidentate_nitrate_fill"
    core_cn = _safe_int(row.get("coreCN"), 0)
    denticity = _safe_int(row.get("DENTATE"), 0)
    n_ligs = _safe_int(row.get("n_ligs"), 0)
    donor_sites = denticity * n_ligs
    open_sites = core_cn - donor_sites
    if core_cn not in {8, 9} or denticity < 1 or n_ligs < 1 or open_sites < 1:
        return None, "invalid_or_nonadjacent_cn_template"
    candidates = []
    for candidate_cn in (8, 9):
        candidate_open = candidate_cn - donor_sites
        if candidate_cn != core_cn and candidate_open >= 0 and candidate_open % 2 < open_sites % 2:
            candidates.append((candidate_open % 2, candidate_cn, candidate_open))
    if not candidates:
        return None, "adjacent_cn_does_not_improve_fill_parity"
    _, proposed_cn, proposed_open = min(candidates)
    proposal = _new_hypothesis_row(
        row,
        root_source_build_id=_safe_text(row.get("root_source_build_id")) or _safe_text(row.get("source_build_id")),
        parent_build_id=_safe_text(row.get("build_id")),
        route="adaptive_cn_fill",
        version=f"{version}:adaptive_cn_fill",
        core_cn=proposed_cn,
        n_fill=proposed_open,
        reason="repeated_ambiguous_shell; adjacent_CN_removes_odd_nitrate_fill_site",
    )
    proposal.update({
        "original_coreCN": core_cn,
        "original_n_fill": open_sites,
        "proposed_coreCN": proposed_cn,
        "proposed_n_fill": proposed_open,
    })
    return proposal, "ok"


def _nitrate_parity_candidate(row: dict, version: str) -> tuple[dict | None, str]:
    core_cn = _safe_int(row.get("coreCN"), 0)
    denticity = _safe_int(row.get("DENTATE"), 0)
    n_ligs = _safe_int(row.get("n_ligs"), 0)
    open_sites = core_cn - denticity * n_ligs
    if (
        _safe_text(row.get("inner_sphere_anion")).lower() != "nitrate"
        or core_cn not in {8, 9}
        or open_sites < 1
        or open_sites % 2 != 1
    ):
        return None, "not_odd_nitrate_fill"
    proposed_cn = 8 if core_cn == 9 else 9
    proposed_open = proposed_cn - denticity * n_ligs
    if proposed_open < 0 or proposed_open % 2:
        return None, "adjacent_cn_does_not_remove_secondary_water"
    proposal = _new_hypothesis_row(
        row,
        root_source_build_id=_safe_text(row.get("root_source_build_id")) or _safe_text(row.get("build_id")),
        parent_build_id=_safe_text(row.get("build_id")),
        route="nitrate_parity_cn",
        version=f"{version}:nitrate_parity_cn",
        core_cn=proposed_cn,
        n_fill=proposed_open,
        reason=(
            "file_QC_NITRATE_MISSING; adjacent_CN replaces unsupported odd "
            "secondary-water fill with an even nitrate-fill occupancy"
        ),
    )
    proposal.update({
        "original_coreCN": core_cn,
        "original_n_fill": open_sites,
        "proposed_coreCN": proposed_cn,
        "proposed_n_fill": proposed_open,
    })
    return proposal, "ok"


ALL_REMAINING_ROUTE_CONFIG = {
    "adaptive_cn_fill": ("standard,large_ligand_fast", 2, 3600, 8, 3, "3G", "12:00:00"),
    "aminopoly_cn8": ("standard,large_ligand_deep,uff_xtb_no_preopt", 3, 5400, 8, 2, "5G", "22:00:00"),
    "placement_qc": ("large_ligand_fast,standard,large_ligand_deep", 3, 5400, 12, 3, "5G", "22:00:00"),
    "placement_build": ("large_ligand_fast,uff_xtb_no_preopt,uff_unrelaxed", 3, 7200, 9, 3, "5G", "23:30:00"),
    "canonical_template_replan": ("large_ligand_fast,uff_xtb_no_preopt,uff_unrelaxed", 3, 7200, 4, 2, "5G", "23:30:00"),
    "canonical_missing": ("large_ligand_fast,uff_xtb_no_preopt,uff_unrelaxed", 3, 7200, 8, 3, "5G", "23:30:00"),
    "nitrate_parity_cn": ("standard,large_ligand_fast,uff_xtb_no_preopt", 3, 5400, 10, 3, "4G", "20:00:00"),
}


def prepare_all_remaining(args) -> int:
    """Build one audited, non-overlapping plan for every unresolved source spec."""
    plan_dir = Path(args.all_remaining_plan_dir)
    runs_root = Path(args.all_remaining_runs_dir)
    out_root = Path(args.all_remaining_out_dir)
    version = _safe_text(args.hypothesis_version, ALL_REMAINING_VERSION)
    remaining_scope = _safe_text(
        getattr(args, "remaining_scope", "known-unfinished"), "known-unfinished",
    )
    specs = _planned_specs(args)
    if specs["build_id"].astype(str).duplicated().any():
        raise SystemExit(f"{args.specs} contains duplicate build_id values")
    baseline_by_id = {str(row["build_id"]): row for _, row in specs.iterrows()}

    index = _read_csv_if_exists(Path(args.geometry_index))
    index_by_id = {
        str(row["build_id"]): row for _, row in index.iterrows()
    } if "build_id" in index.columns else {}
    ligtype_args = argparse.Namespace(
        ligtype_override_index=load_ligtype_overrides(args.ligtype_overrides)
    )

    family_plan = Path(args.family_plan_dir)
    family_runs = Path(args.family_runs_dir)
    config_path = family_plan / "run_config.csv"
    family_config = _read_csv_if_exists(config_path)
    family_spec_by_build: dict[str, dict] = {}
    root_by_build: dict[str, str] = {}
    group_by_root: dict[str, str] = {}
    failed_by_build: dict[str, dict] = {}
    attempts_by_build: dict[str, list[dict]] = {}
    accepted_candidates_by_root: dict[str, list[tuple[str, str, dict]]] = {}

    for config_row in family_config.to_dict("records"):
        group = _safe_text(config_row.get("run_group"))
        queue_path = _queue_path_from_config(config_row.get("queue"), family_plan, group)
        queue = _read_csv_if_exists(queue_path)
        report_dir = family_runs / group
        accepted = _regeneration_report_union(report_dir, REGENERATE_ACCEPTED_NAME)
        failed = _regeneration_report_union(report_dir, REGENERATE_STILL_FAILED_NAME)
        attempts = _deduplicate_regeneration_attempts(
            _regeneration_report_union(report_dir, REGENERATE_ATTEMPTS_NAME)
        )
        for queue_row in queue.to_dict("records"):
            build_id = _safe_text(queue_row.get("build_id"))
            root = _safe_text(queue_row.get("root_source_build_id")) or _safe_text(
                queue_row.get("source_build_id")
            ) or build_id
            if root in group_by_root and group_by_root[root] != group:
                raise SystemExit(f"Family source {root} appears in multiple groups")
            root_by_build[build_id] = root
            group_by_root[root] = group
            family_spec_by_build[build_id] = queue_row
        for row in failed.to_dict("records"):
            failed_by_build[_safe_text(row.get("build_id"))] = row
        for row in attempts.to_dict("records"):
            attempts_by_build.setdefault(_safe_text(row.get("build_id")), []).append(row)
        for row in accepted.to_dict("records"):
            build_id = _safe_text(row.get("build_id"))
            root = root_by_build.get(build_id)
            spec = family_spec_by_build.get(build_id)
            if root and spec:
                accepted_candidates_by_root.setdefault(root, []).append((
                    build_id, _safe_text(row.get(gschema.ACCEPTED_XYZ_PATH)), spec,
                ))

    adaptive_queue = _read_csv_if_exists(Path(args.adaptive_output))
    adaptive_build_ids = set(
        adaptive_queue["build_id"].astype(str)
    ) if "build_id" in adaptive_queue.columns else set()
    adaptive_report_dir = family_runs / "adaptive_cn_fill"
    adaptive_accepted = _regeneration_report_union(adaptive_report_dir, REGENERATE_ACCEPTED_NAME)
    adaptive_failed = _regeneration_report_union(adaptive_report_dir, REGENERATE_STILL_FAILED_NAME)
    adaptive_attempts = _deduplicate_regeneration_attempts(
        _regeneration_report_union(adaptive_report_dir, REGENERATE_ATTEMPTS_NAME)
    )
    for row in adaptive_queue.to_dict("records"):
        build_id = _safe_text(row.get("build_id"))
        parent = _safe_text(row.get("parent_build_id")) or _safe_text(row.get("source_build_id"))
        root = _safe_text(row.get("root_source_build_id")) or root_by_build.get(parent, parent)
        root_by_build[build_id] = root
        family_spec_by_build[build_id] = row
    for row in adaptive_failed.to_dict("records"):
        failed_by_build[_safe_text(row.get("build_id"))] = row
    for row in adaptive_attempts.to_dict("records"):
        attempts_by_build.setdefault(_safe_text(row.get("build_id")), []).append(row)
    for row in adaptive_accepted.to_dict("records"):
        build_id = _safe_text(row.get("build_id"))
        root = root_by_build.get(build_id)
        spec = family_spec_by_build.get(build_id)
        if root and spec:
            accepted_candidates_by_root.setdefault(root, []).append((
                build_id, _safe_text(row.get(gschema.ACCEPTED_XYZ_PATH)), spec,
            ))

    main_accepted: dict[str, list[str]] = {}
    for report_dir in (Path(args.historical_reports_dir), Path(args.historical_reports_dir) / "missing_geometry_rescue"):
        accepted = _regeneration_report_union(report_dir, REGENERATE_ACCEPTED_NAME)
        for row in accepted.to_dict("records"):
            main_accepted.setdefault(_safe_text(row.get("build_id")), []).append(
                _safe_text(row.get(gschema.ACCEPTED_XYZ_PATH))
            )

    queues: dict[str, list[dict]] = {route: [] for route in ALL_REMAINING_ROUTE_CONFIG}
    manifest: list[dict] = []
    deferred_existing_successes = 0
    accepted_aminopoly_cn8_precedents: list[dict] = []
    for root, candidates in accepted_candidates_by_root.items():
        if group_by_root.get(root) != "template_aminopolycarboxylate":
            continue
        for build_id, path, spec in candidates:
            result = _dual_qc_candidate(path, spec, args)
            if result["accepted"] and _safe_int(spec.get("coreCN"), 0) == 8:
                accepted_aminopoly_cn8_precedents.append({
                    "root_source_build_id": root,
                    "build_id": build_id,
                    "xyz_path": result["xyz_path"],
                })

    def queue_manifest(root: str, baseline_id: str, proposal: dict, source_state: str) -> None:
        route = _safe_text(proposal.get("rescue_route"))
        if route not in queues:
            raise SystemExit(f"Unsupported all-remaining route {route!r}")
        proposal["root_source_build_id"] = root
        proposal["source_build_id"] = root
        proposal["qc_class"] = _safe_text(proposal.get("best_qc_class")) or _safe_text(
            proposal.get("qc_class")
        ) or source_state
        queues[route].append(proposal)
        manifest.append({
            "root_source_build_id": root,
            "baseline_build_id": baseline_id,
            "state": "queued",
            "route": route,
            "planned_build_id": _safe_text(proposal.get("build_id")),
            "parent_build_id": _safe_text(proposal.get("parent_build_id")),
            "source_state": source_state,
            "resolution_build_id": "",
            "resolution_xyz_path": "",
            "note": _safe_text(proposal.get("hypothesis_reason")),
        })

    for baseline_id, baseline_spec in baseline_by_id.items():
        root = baseline_id
        if root in group_by_root:
            accepted_result = None
            accepted_build_id = ""
            rejected_candidate_results: list[tuple[str, dict, dict]] = []
            for build_id, path, candidate_spec in accepted_candidates_by_root.get(root, []):
                result = _dual_qc_candidate(path, candidate_spec, args)
                if result["accepted"]:
                    accepted_result = result
                    accepted_build_id = build_id
                    break
                if result["candidate_exists"]:
                    rejected_candidate_results.append((build_id, result, candidate_spec))
            if accepted_result is not None:
                if remaining_scope == "strict-baseline-audit":
                    manifest.append({
                        "root_source_build_id": root,
                        "baseline_build_id": baseline_id,
                        "state": "resolved_existing",
                        "route": "",
                        "planned_build_id": "",
                        "parent_build_id": "",
                        "source_state": "family_regeneration",
                        "resolution_build_id": accepted_build_id,
                        "resolution_xyz_path": accepted_result["xyz_path"],
                        "resolution_spec_json": _qc_spec_snapshot(candidate_spec),
                        "note": "historical family/adaptive candidate passed current dual QC",
                    })
                else:
                    deferred_existing_successes += 1
                continue

            descendants = [build_id for build_id, source in root_by_build.items() if source == root]
            failed_descendants = [build_id for build_id in descendants if build_id in failed_by_build]
            if failed_descendants:
                adaptive_failed_descendants = [
                    build_id for build_id in failed_descendants
                    if build_id in adaptive_build_ids
                ]
                current_build = (
                    adaptive_failed_descendants[-1]
                    if adaptive_failed_descendants else failed_descendants[-1]
                )
                current_spec = dict(family_spec_by_build[current_build])
                failure = dict(failed_by_build[current_build])
                current = {**current_spec, **failure}
            elif rejected_candidate_results:
                # Historical reports used nearest-core QC alone.  Re-audit any
                # such "accepted" candidate through file/composition QC and
                # keep the real artifact as the diagnosed parent hypothesis.
                current_build, rejected_result, current_spec = min(
                    rejected_candidate_results,
                    key=lambda item: _best_attempt_rank({
                        "qc_class": item[1].get("qc_class"),
                        "coreCN_max_dist": item[1].get("coreCN_max_dist"),
                        "gap_after_coreCN": item[1].get("gap_after_coreCN"),
                    }),
                )
                current = {
                    **dict(current_spec),
                    "best_qc_class": _safe_text(rejected_result.get("qc_class")),
                    "best_file_qc_status": _safe_text(rejected_result.get("file_qc_status")),
                    "best_coreCN_max_dist": rejected_result.get("coreCN_max_dist", ""),
                    "best_gap_after_coreCN": rejected_result.get("gap_after_coreCN", ""),
                    "best_xyz_path": _safe_text(rejected_result.get("xyz_path")),
                    "failure_note": _safe_text(rejected_result.get("file_qc_note"))
                    or _safe_text(rejected_result.get("qc_note")),
                }
            else:
                manifest.append({
                    "root_source_build_id": root,
                    "baseline_build_id": baseline_id,
                    "state": "blocked_unaccounted",
                    "route": "",
                    "planned_build_id": "",
                    "parent_build_id": "",
                    "source_state": "family_regeneration",
                    "resolution_build_id": "",
                    "resolution_xyz_path": "",
                    "note": "no accepted candidate and no terminal family failure report",
                })
                continue
            current.update({
                "root_source_build_id": root,
                "source_build_id": root,
                "parent_build_id": current_build,
                "build_id": current_build,
                "ambiguous_attempts": sum(
                    _safe_text(row.get("qc_class")) == "BORDERLINE_AMBIGUOUS_SHELL"
                    for row in attempts_by_build.get(current_build, [])
                ),
                "candidate_exists": bool(_candidate_xyz_path(current.get("best_xyz_path"), current)),
            })
            if _safe_text(current.get("best_file_qc_status")) == "NITRATE_MISSING":
                proposal, reason = _nitrate_parity_candidate(current, version)
                if proposal is None:
                    manifest.append({
                        "root_source_build_id": root,
                        "baseline_build_id": baseline_id,
                        "state": "blocked_chemistry",
                        "route": "",
                        "planned_build_id": "",
                        "parent_build_id": current_build,
                        "source_state": "NITRATE_MISSING",
                        "resolution_build_id": "",
                        "resolution_xyz_path": "",
                        "note": reason,
                    })
                    continue
                proposal["ligtype_sequence"] = _alternative_ligtype_sequence(
                    proposal, _safe_text(current.get("ligtype_override")),
                    include_current=True, max_candidates=3,
                )
                queue_manifest(root, baseline_id, proposal, "NITRATE_MISSING")
                continue
            adaptive, _ = _adaptive_cn_fill_candidate(
                current, version, float(args.long_bond_threshold)
            )
            if adaptive is not None and _safe_text(adaptive.get("build_id")) not in root_by_build:
                current_ligtype = _safe_text(current.get("ligtype_override"))
                adaptive["ligtype_sequence"] = _alternative_ligtype_sequence(
                    adaptive, current_ligtype, include_current=True, max_candidates=2,
                )
                queue_manifest(root, baseline_id, adaptive, _safe_text(current.get("best_qc_class")))
                continue

            best_qc = _safe_text(current.get("best_qc_class"))
            no_candidate = not bool(current.get("candidate_exists"))
            if (
                group_by_root[root] == "template_aminopolycarboxylate"
                and no_candidate
                and _safe_int(current.get("coreCN"), 0) == 9
                and _safe_int(current.get("DENTATE"), 0) == 6
                and _safe_int(current.get("n_ligs"), 0) == 1
                and bool(accepted_aminopoly_cn8_precedents)
            ):
                precedent_build_ids = sorted({
                    _safe_text(item.get("build_id"))
                    for item in accepted_aminopoly_cn8_precedents
                    if _safe_text(item.get("build_id"))
                })
                proposal = _new_hypothesis_row(
                    current,
                    root_source_build_id=root,
                    parent_build_id=current_build,
                    route="aminopoly_cn8",
                    version=f"{version}:aminopoly_cn8",
                    core_cn=8,
                    n_fill=2,
                    reason=(
                        "CN9 no-structures with odd nitrate fill; dual-QC-accepted "
                        "aminopolycarboxylate family precedent exists at CN8"
                    ),
                )
                proposal["family_precedent_build_ids"] = json.dumps(precedent_build_ids)
                proposal["ligtype_sequence"] = _alternative_ligtype_sequence(
                    proposal, _safe_text(current.get("ligtype_override")),
                    include_current=True, max_candidates=3,
                )
                queue_manifest(root, baseline_id, proposal, "BUILD_FAILED")
                continue

            route = "placement_build" if no_candidate or best_qc in {"", "BUILD_FAILED"} else "placement_qc"
            proposal = dict(current)
            proposal.update({
                "root_source_build_id": root,
                "source_build_id": root,
                "parent_build_id": current_build,
                "rescue_route": route,
                "hypothesis_version": f"{version}:{route}",
                "hypothesis_reason": (
                    "new placement hypothesis after exhausted build profiles"
                    if route == "placement_build"
                    else "alternate placement class after repeated geometry-QC diagnosis"
                ),
                "ligtype_sequence": _alternative_ligtype_sequence(
                    proposal, _safe_text(current.get("ligtype_override")),
                    include_current=False, max_candidates=3,
                ),
            })
            queue_manifest(root, baseline_id, proposal, best_qc or "BUILD_FAILED")
            continue

        # Non-family baseline: prefer any historical regenerated candidate, then
        # the canonical geometry-index path. Every acceptance is re-audited.
        index_row_data = index_by_id.get(baseline_id, {})
        status = _safe_text(index_row_data.get("status"), "missing_index_row")
        if remaining_scope == "known-unfinished" and status in SUCCESS_STATUSES:
            # The default rescue is intentionally bounded to work already known
            # to be unfinished.  Retrospectively invalidating every historical
            # success is a separate, explicit audit because it can multiply the
            # queue by hundreds of expensive builds.
            deferred_existing_successes += 1
            continue
        candidates = list(main_accepted.get(baseline_id, []))
        if _safe_text(index_row_data.get("xyz_path")):
            candidates.append(_safe_text(index_row_data.get("xyz_path")))
        accepted_result = None
        last_result = None
        for path in candidates:
            result = _dual_qc_candidate(path, baseline_spec, args)
            last_result = result
            if result["accepted"]:
                accepted_result = result
                break
        if accepted_result is not None:
            manifest.append({
                "root_source_build_id": root,
                "baseline_build_id": baseline_id,
                "state": "resolved_existing",
                "route": "",
                "planned_build_id": "",
                "parent_build_id": "",
                "source_state": _safe_text(index_row_data.get("status"), "planned"),
                "resolution_build_id": baseline_id,
                "resolution_xyz_path": accepted_result["xyz_path"],
                "resolution_spec_json": _qc_spec_snapshot(baseline_spec),
                "note": "historical candidate passed current dual QC",
            })
            continue

        base = dict(baseline_spec)
        base.update({
            "root_source_build_id": root,
            "source_build_id": root,
            "parent_build_id": baseline_id,
            "source_xyz_path": (last_result or {}).get("xyz_path", ""),
            "source_failure_class": (last_result or {}).get("qc_class", status),
            "candidate_exists": bool((last_result or {}).get("candidate_exists", False)),
        })
        if status in SUCCESS_STATUSES and not bool(
            (last_result or {}).get("candidate_exists", False)
        ):
            manifest.append({
                "root_source_build_id": root,
                "baseline_build_id": baseline_id,
                "state": "blocked_unavailable",
                "route": "",
                "planned_build_id": "",
                "parent_build_id": baseline_id,
                "source_state": status,
                "resolution_build_id": "",
                "resolution_xyz_path": "",
                "note": "successful index row has no accessible XYZ; run planning on cluster storage",
            })
            continue

        failure_class = _safe_text((last_result or {}).get("file_qc_status"))
        if failure_class == "NITRATE_MISSING" or _safe_text(index_row_data.get("qc_status")) == "NITRATE_MISSING":
            proposal, reason = _nitrate_parity_candidate(base, version)
            if proposal is None:
                manifest.append({
                    "root_source_build_id": root,
                    "baseline_build_id": baseline_id,
                    "state": "blocked_chemistry",
                    "route": "",
                    "planned_build_id": "",
                    "parent_build_id": baseline_id,
                    "source_state": "NITRATE_MISSING",
                    "resolution_build_id": "",
                    "resolution_xyz_path": "",
                    "note": reason,
                })
                continue
            current_ligtype = _ligtype_override_for_spec(pd.Series(base), ligtype_args)
            proposal["ligtype_sequence"] = _alternative_ligtype_sequence(
                proposal, current_ligtype, include_current=True, max_candidates=3,
            )
            queue_manifest(root, baseline_id, proposal, "NITRATE_MISSING")
            continue

        if status == "failed_no_structures":
            smiles = _safe_text(base.get("SMILES_FOR_ARCHITECTOR"))
            donors = detect_donors(smiles)
            values = _family_template_values(base, smiles, donors) if donors is not None else None
            if (
                donors is not None and donors.strategy == "compact_amide_core"
                and values is not None and values["template_changed"]
                and not _selected_azole_is_ambiguous(smiles, donors.coord_list)
            ):
                proposal = dict(base)
                proposal.update({
                    "COORDLIST": json.dumps(donors.coord_list),
                    "DONOR_TYPES": json.dumps(donors.donor_types),
                    "DENTATE": donors.denticity,
                    "n_ligs": values["proposed_n_ligs"],
                })
                proposal = _new_hypothesis_row(
                    proposal,
                    root_source_build_id=root,
                    parent_build_id=baseline_id,
                    route="canonical_template_replan",
                    version=f"{version}:canonical_template_replan",
                    core_cn=values["proposed_core_cn"],
                    n_fill=values["proposed_n_fill"],
                    reason="curated compact-amide template correction after canonical no-structures",
                )
                proposal["ligtype_sequence"] = _alternative_ligtype_sequence(
                    proposal, _suggest_ligtype(donors), include_current=True, max_candidates=3,
                )
            else:
                proposal = dict(base)
                current_ligtype = _ligtype_override_for_spec(pd.Series(base), ligtype_args)
                proposal.update({
                    "rescue_route": "canonical_missing",
                    "hypothesis_version": f"{version}:canonical_missing",
                    "hypothesis_reason": "canonical no-structures; alternate placement classes",
                    "ligtype_sequence": _alternative_ligtype_sequence(
                        proposal, current_ligtype, include_current=False, max_candidates=3,
                    ),
                })
            queue_manifest(root, baseline_id, proposal, status)
            continue

        if last_result is not None and last_result.get("candidate_exists"):
            proposal = dict(base)
            current_ligtype = _ligtype_override_for_spec(pd.Series(base), ligtype_args)
            proposal.update({
                "rescue_route": "placement_qc",
                "hypothesis_version": f"{version}:placement_qc",
                "hypothesis_reason": "baseline candidate failed current dual QC",
                "best_qc_class": _safe_text(last_result.get("qc_class")),
                "best_xyz_path": _safe_text(last_result.get("xyz_path")),
                "ligtype_sequence": _alternative_ligtype_sequence(
                    proposal, current_ligtype, include_current=False, max_candidates=3,
                ),
            })
            queue_manifest(root, baseline_id, proposal, _safe_text(last_result.get("qc_class")))
            continue

        proposal = dict(base)
        current_ligtype = _ligtype_override_for_spec(pd.Series(base), ligtype_args)
        proposal.update({
            "rescue_route": "canonical_missing",
            "hypothesis_version": f"{version}:canonical_missing",
            "hypothesis_reason": f"no accepted accessible candidate; source status={status}",
            "ligtype_sequence": _alternative_ligtype_sequence(
                proposal, current_ligtype, include_current=False, max_candidates=3,
            ),
        })
        queue_manifest(root, baseline_id, proposal, status)

    for row in manifest:
        row.setdefault("resolution_spec_json", "")
    manifest_frame = pd.DataFrame(manifest)
    if manifest_frame.empty:
        manifest_frame = pd.DataFrame(columns=[
            "root_source_build_id", "baseline_build_id", "state", "route",
            "planned_build_id", "parent_build_id", "source_state",
            "resolution_build_id", "resolution_xyz_path", "note",
            "resolution_spec_json",
        ])
    if manifest_frame["root_source_build_id"].astype(str).duplicated().any():
        raise SystemExit("All-remaining manifest contains duplicate root_source_build_id values")
    queued = manifest_frame[manifest_frame["state"].astype(str).eq("queued")]
    if queued["planned_build_id"].astype(str).duplicated().any():
        duplicated = sorted(set(
            queued.loc[queued["planned_build_id"].astype(str).duplicated(keep=False), "planned_build_id"].astype(str)
        ))
        raise SystemExit(f"All-remaining queues collide on build_id: {duplicated[:10]}")

    plan_dir.mkdir(parents=True, exist_ok=True)
    config_rows = []
    config_by_route: dict[str, dict] = {}
    for route, values in ALL_REMAINING_ROUTE_CONFIG.items():
        frame = pd.DataFrame(queues[route])
        queue_path = plan_dir / f"{route}.csv"
        if frame.empty:
            frame = pd.DataFrame(columns=list(dict.fromkeys([
                *specs.columns, "root_source_build_id", "source_build_id", "parent_build_id",
                "rescue_route", "hypothesis_version", "hypothesis_reason", "ligtype_sequence",
            ])))
        frame = frame.sort_values("root_source_build_id", kind="stable").reset_index(drop=True)
        _write_csv_atomic(frame, queue_path)
        queue_hash = _sha256_file(queue_path)
        profiles, attempts, timeout, shard_cap, concurrency, memory, walltime = values
        rows = len(frame)
        num_shards = min(int(shard_cap), rows) if rows else 0
        max_concurrent = min(int(concurrency), num_shards) if num_shards else 0
        run_id = f"{version}:{route}:{queue_hash[:12]}"
        config_row = {
            "run_group": route,
            "queue": _portable_path(queue_path),
            "rows": rows,
            "profile_sequence": profiles,
            "max_attempts": attempts,
            "timeout_per_complex": timeout,
            "num_shards": num_shards,
            "max_concurrent_tasks": max_concurrent,
            "memory": memory,
            "walltime": walltime,
            "run_id": run_id,
            "queue_sha256": queue_hash,
            "reports_dir": _portable_path(runs_root / route),
            "regen_out": _portable_path(out_root / route),
            "remaining_scope": remaining_scope,
        }
        config_rows.append(config_row)
        config_by_route[route] = config_row
    config_frame = pd.DataFrame(config_rows)
    _write_csv_atomic(config_frame, plan_dir / "run_config.csv")

    for row in manifest:
        route = _safe_text(row.get("route"))
        config_row = config_by_route.get(route)
        if config_row:
            row["run_id"] = config_row["run_id"]
            row["queue_sha256"] = config_row["queue_sha256"]
            row["reports_dir"] = config_row["reports_dir"]
            row["regen_out"] = config_row["regen_out"]
        else:
            row.update({"run_id": "", "queue_sha256": "", "reports_dir": "", "regen_out": ""})
    if manifest:
        manifest_frame = pd.DataFrame(manifest)
    _write_csv_atomic(manifest_frame, plan_dir / "manifest.csv")

    state_counts = manifest_frame["state"].value_counts().to_dict()
    route_counts = queued["route"].value_counts().to_dict()
    blocked_count = (
        len(manifest_frame)
        - state_counts.get("resolved_existing", 0)
        - state_counts.get("queued", 0)
    )
    coverage_total = deferred_existing_successes + len(manifest_frame)
    if coverage_total != len(specs):
        raise SystemExit(
            "All-remaining coverage invariant failed: "
            f"deferred={deferred_existing_successes} + manifest={len(manifest_frame)} "
            f"!= baseline={len(specs)}"
        )
    plan_meta = {
        "schema_version": 1,
        "hypothesis_version": version,
        "remaining_scope": remaining_scope,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_count": len(specs),
        "deferred_existing_successes": deferred_existing_successes,
        "manifest_count": len(manifest_frame),
        "queued_count": int(state_counts.get("queued", 0)),
        "resolved_existing_count": int(state_counts.get("resolved_existing", 0)),
        "blocked_count": int(blocked_count),
        "coverage_total": int(coverage_total),
        "specs": _portable_path(Path(args.specs)),
        "specs_sha256": _sha256_file(Path(args.specs)),
        "geometry_index": _portable_path(Path(args.geometry_index)),
        "geometry_index_sha256": _sha256_file(Path(args.geometry_index)),
        "family_run_config": _portable_path(config_path),
        "family_run_config_sha256": _sha256_file(config_path),
        "adaptive_queue": _portable_path(Path(args.adaptive_output)),
        "adaptive_queue_sha256": _sha256_file(Path(args.adaptive_output)),
        "manifest_sha256": _sha256_file(plan_dir / "manifest.csv"),
        "run_config_sha256": _sha256_file(plan_dir / "run_config.csv"),
    }
    _write_json_atomic(plan_meta, plan_dir / ALL_REMAINING_PLAN_META_NAME)
    summary_lines = [
        "All-remaining regeneration plan",
        "",
        f"Hypothesis version: {version}",
        f"Scope: {remaining_scope}",
        f"Baseline source specs: {len(specs)}",
        f"Deferred existing successes: {deferred_existing_successes}",
        f"Targeted sources in manifest: {len(manifest_frame)}",
        f"Resolved by current dual QC: {state_counts.get('resolved_existing', 0)}",
        f"Queued: {state_counts.get('queued', 0)}",
        f"Blocked/unaccounted: {blocked_count}",
        "",
        "Queued routes:",
        json.dumps(route_counts, indent=2, sort_keys=True),
        "",
        "Acceptance requires exact heavy-element composition and nearest-coreCN QC == OK.",
        "Original canonical and prior regenerated geometries are never overwritten.",
    ]
    (plan_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("All-remaining regeneration plan")
    print(f"  scope: {remaining_scope}")
    print(f"  baseline source specs: {len(specs)}")
    print(f"  deferred existing successes: {deferred_existing_successes}")
    print(f"  targeted sources in manifest: {len(manifest_frame)}")
    print(f"  resolved existing: {state_counts.get('resolved_existing', 0)}")
    print(f"  queued: {state_counts.get('queued', 0)}")
    for route, count in sorted(route_counts.items()):
        print(f"    {route}: {count}")
    print(f"  blocked/unaccounted: {blocked_count}")
    print(f"  manifest: {plan_dir / 'manifest.csv'}")
    print(f"  run config: {plan_dir / 'run_config.csv'}")
    print(f"  plan metadata: {plan_dir / ALL_REMAINING_PLAN_META_NAME}")
    return 2 if blocked_count else 0


def all_remaining_status(args) -> int:
    """Reconcile an immutable all-remaining manifest to strict final states."""
    plan_dir = Path(args.all_remaining_plan_dir)
    runs_root = Path(args.all_remaining_runs_dir)
    manifest_path = plan_dir / "manifest.csv"
    config_path = plan_dir / "run_config.csv"
    plan_meta_path = plan_dir / ALL_REMAINING_PLAN_META_NAME
    if not manifest_path.exists() or not config_path.exists() or not plan_meta_path.exists():
        print(f"Missing all-remaining plan in {plan_dir}")
        return 2
    try:
        plan_meta = json.loads(plan_meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Invalid all-remaining plan metadata {plan_meta_path}: {exc}")
        return 2
    manifest = _read_csv_if_exists(manifest_path)
    config = _read_csv_if_exists(config_path)
    plan_errors: list[str] = []
    if _safe_text(plan_meta.get("manifest_sha256")) != _sha256_file(manifest_path):
        plan_errors.append("manifest_sha256_mismatch")
    if _safe_text(plan_meta.get("run_config_sha256")) != _sha256_file(config_path):
        plan_errors.append("run_config_sha256_mismatch")
    baseline_count = _safe_int(plan_meta.get("baseline_count"), -1)
    deferred_count = _safe_int(plan_meta.get("deferred_existing_successes"), -1)
    manifest_count = _safe_int(plan_meta.get("manifest_count"), -1)
    if manifest_count != len(manifest):
        plan_errors.append("manifest_count_mismatch")
    if baseline_count < 0 or deferred_count < 0 or deferred_count + manifest_count != baseline_count:
        plan_errors.append("baseline_coverage_invariant_failed")
    if "root_source_build_id" in manifest.columns and manifest["root_source_build_id"].astype(str).duplicated().any():
        plan_errors.append("duplicate_manifest_root_source_build_id")
    if "run_group" not in config.columns or config["run_group"].astype(str).duplicated().any():
        plan_errors.append("invalid_or_duplicate_run_group")
    planned_queued = int(
        manifest["state"].astype(str).eq("queued").sum()
        if "state" in manifest.columns else 0
    )
    configured_rows = int(
        sum(_safe_int(value, 0) for value in config.get("rows", pd.Series(dtype=int)))
    )
    if planned_queued != configured_rows or planned_queued != _safe_int(plan_meta.get("queued_count"), -1):
        plan_errors.append("queued_count_invariant_failed")
    if plan_errors:
        print("All-remaining plan integrity check failed:")
        for error in plan_errors:
            print(f"  {error}")
        return 2

    queue_specs: dict[str, dict] = {}
    accepted_by_route: dict[str, set[str]] = {}
    failed_by_route: dict[str, set[str]] = {}
    invalid_by_route: dict[str, set[str]] = {}
    integrity_error_routes: set[str] = set()
    for config_row in config.to_dict("records"):
        route = _safe_text(config_row.get("run_group"))
        queue_path = Path(_safe_text(config_row.get("queue")))
        if not queue_path.is_absolute():
            queue_path = _REPO_ROOT / queue_path
        queue_hash = _safe_text(config_row.get("queue_sha256"))
        if _sha256_file(queue_path) != queue_hash:
            integrity_error_routes.add(route)
            accepted_by_route[route] = set()
            failed_by_route[route] = set()
            invalid_by_route[route] = set()
            continue
        queue = _read_csv_if_exists(queue_path)
        if len(queue) != _safe_int(config_row.get("rows"), -1):
            integrity_error_routes.add(route)
            accepted_by_route[route] = set()
            failed_by_route[route] = set()
            invalid_by_route[route] = set()
            continue
        if queue.empty:
            accepted_by_route[route] = set()
            failed_by_route[route] = set()
            invalid_by_route[route] = set()
            continue
        for row in queue.to_dict("records"):
            queue_specs[_safe_text(row.get("build_id"))] = row
        report_dir = Path(_safe_text(config_row.get("reports_dir")))
        if not report_dir.is_absolute():
            report_dir = _REPO_ROOT / report_dir
        run_id = _safe_text(config_row.get("run_id"))
        num_shards = _safe_int(config_row.get("num_shards"), 0)
        expected_strategy = ""
        meta_payloads = []
        for expected_shard_id, meta_path in enumerate(_regen_shard_report_paths(
            report_dir, REGENERATE_RUN_META_NAME, num_shards,
        )):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_payloads = []
                break
            if not all((
                _safe_text(payload.get("run_id")) == run_id,
                _safe_text(payload.get("queue_sha256")) == queue_hash,
                _safe_int(payload.get("num_shards"), -1) == num_shards,
                _safe_int(payload.get("shard_id"), -1) == expected_shard_id,
                bool(_safe_text(payload.get("strategy_sha256"))),
            )):
                meta_payloads = []
                break
            meta_payloads.append(payload)
        strategy_values = {
            _safe_text(payload.get("strategy_sha256")) for payload in meta_payloads
        }
        if len(meta_payloads) == num_shards and len(strategy_values) == 1:
            expected_strategy = next(iter(strategy_values))

        # Final reconciliation consumes only the canonical reports emitted by
        # a strict merge.  Raw shard reports from a previous strategy must not
        # turn a pending immutable run into a false acceptance.
        accepted = _read_csv_if_exists(report_dir / REGENERATE_ACCEPTED_NAME)
        failed = _read_csv_if_exists(report_dir / REGENERATE_STILL_FAILED_NAME)
        required_fingerprint = {"run_id", "queue_sha256", "strategy_sha256"}
        for name, frame in (("accepted", accepted), ("failed", failed)):
            if not expected_strategy or not required_fingerprint.issubset(frame.columns):
                filtered = frame.iloc[0:0].copy()
            else:
                filtered = frame[
                    frame["run_id"].astype(str).eq(run_id)
                    & frame["queue_sha256"].astype(str).eq(queue_hash)
                    & frame["strategy_sha256"].astype(str).eq(expected_strategy)
                ].copy()
            if name == "accepted":
                accepted = filtered
            else:
                failed = filtered
        accepted_ids: set[str] = set()
        invalid_ids: set[str] = set()
        for row in accepted.to_dict("records"):
            build_id = _safe_text(row.get("build_id"))
            spec = queue_specs.get(build_id)
            if spec is None:
                invalid_ids.add(build_id)
                continue
            result = _dual_qc_candidate(row.get(gschema.ACCEPTED_XYZ_PATH), spec, args)
            (accepted_ids if result["accepted"] else invalid_ids).add(build_id)
        failed_ids = set(failed["build_id"].astype(str)) if "build_id" in failed.columns else set()
        accepted_by_route[route] = accepted_ids
        failed_by_route[route] = failed_ids - accepted_ids
        invalid_by_route[route] = invalid_ids - accepted_ids

    status_rows = []
    for row in manifest.to_dict("records"):
        state = _safe_text(row.get("state"))
        if state == "resolved_existing":
            try:
                resolution_spec = json.loads(_safe_text(row.get("resolution_spec_json")))
            except Exception:
                resolution_spec = None
            if resolution_spec is None:
                final_state = "invalid_resolved_existing"
            else:
                result = _dual_qc_candidate(
                    row.get("resolution_xyz_path"), resolution_spec, args,
                )
                final_state = "accepted" if result["accepted"] else "invalid_resolved_existing"
        elif state != "queued":
            final_state = state
        else:
            route = _safe_text(row.get("route"))
            build_id = _safe_text(row.get("planned_build_id"))
            if route in integrity_error_routes:
                final_state = "integrity_error"
            elif build_id in accepted_by_route.get(route, set()):
                final_state = "accepted"
            elif build_id in invalid_by_route.get(route, set()):
                final_state = "invalid_accepted_report"
            elif build_id in failed_by_route.get(route, set()):
                final_state = "scientifically_rejected"
            else:
                final_state = "pending_or_missing_report"
        out = dict(row)
        out["final_state"] = final_state
        status_rows.append(out)
    status = (
        pd.DataFrame(status_rows)
        if status_rows
        else pd.DataFrame(columns=[*manifest.columns, "final_state"])
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    status_path = runs_root / "all_remaining_status.csv"
    _write_csv_atomic(status, status_path)
    counts = status["final_state"].value_counts().to_dict() if not status.empty else {}
    summary_path = runs_root / "all_remaining_status_summary.txt"
    summary_path.write_text(
        "All-remaining final status\n\n"
        f"Baseline sources: {baseline_count}\n"
        f"Deferred existing successes: {deferred_count}\n"
        f"Targeted manifest sources: {manifest_count}\n\n"
        + json.dumps(counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("All-remaining final status")
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"Status CSV: {status_path}")
    pending_states = {
        "pending_or_missing_report", "invalid_accepted_report",
        "invalid_resolved_existing", "integrity_error",
    }
    if any(state in counts for state in pending_states) or any(
        state.startswith("blocked_") for state in counts
    ):
        return 2
    if counts.get("scientifically_rejected", 0):
        return 3
    return 0


def prepare_hard10_rescue(args) -> int:
    """Freeze the ten job-5981236 failures into two ordered five-row queues."""
    specs = _planned_specs(args)
    if "build_id" not in specs.columns:
        raise SystemExit(f"{args.specs} is missing build_id")
    build_ids = specs["build_id"].astype(str)
    duplicate_ids = sorted(set(build_ids[build_ids.duplicated(keep=False)]))
    targeted = set(HARD10_NO_STRUCTURES_BUILD_IDS) | set(HARD10_NATIVE_CRASH_BUILD_IDS)
    targeted_duplicates = sorted(targeted & set(duplicate_ids))
    if targeted_duplicates:
        raise SystemExit(f"Duplicate hard10 build_id values in {args.specs}: {targeted_duplicates}")

    by_id = {str(row["build_id"]): row for _, row in specs.iterrows()}
    missing_ids = sorted(targeted - set(by_id))
    if missing_ids:
        raise SystemExit(f"Hard10 build_id values absent from {args.specs}: {missing_ids}")

    args.ligtype_override_index = load_ligtype_overrides(args.ligtype_overrides)

    def queue_for(group: str, ordered_ids: tuple[str, ...]) -> pd.DataFrame:
        rows = []
        for build_id in ordered_ids:
            row = by_id[build_id].copy()
            ligtype = _ligtype_override_for_spec(row, args)
            if not ligtype:
                raise SystemExit(
                    f"Hard10 build_id={build_id} lacks an explicit ligType override in "
                    f"{args.ligtype_overrides}"
                )
            row["rescue_group"] = group
            row["rescue_ligtype_override"] = ligtype
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)

    no_structures = queue_for("failed_no_structures", HARD10_NO_STRUCTURES_BUILD_IDS)
    native_crash = queue_for("failed_native_crash", HARD10_NATIVE_CRASH_BUILD_IDS)
    no_structures_output = Path(args.hard10_no_structures_queue_output)
    native_crash_output = Path(args.hard10_native_crash_queue_output)
    _write_csv_atomic(no_structures, no_structures_output)
    _write_csv_atomic(native_crash, native_crash_output)

    print("Hard10 rescue queues prepared")
    print(f"failed_no_structures: {len(no_structures)} -> {no_structures_output}")
    print(f"failed_native_crash: {len(native_crash)} -> {native_crash_output}")
    print("Each queue is ordered for five shards with one molecule per shard.")
    return 0


def regenerate_failed(args) -> int:
    if not Path(args.regenerate_input).exists():
        print(f"Missing regeneration input: {args.regenerate_input}")
        return 1

    if not (0 <= int(args.shard_id) < int(args.num_shards)):
        raise SystemExit("--shard-id must be in [0, num-shards-1]")

    queue_path = Path(args.regenerate_input)
    actual_queue_sha256 = _sha256_file(queue_path)
    requested_queue_sha256 = _safe_text(getattr(args, "queue_sha256", ""))
    if requested_queue_sha256 and requested_queue_sha256 != actual_queue_sha256:
        raise SystemExit(
            "Regeneration queue fingerprint mismatch: "
            f"expected={requested_queue_sha256}, actual={actual_queue_sha256}"
        )
    args.queue_sha256 = actual_queue_sha256
    args.strategy_sha256 = _regeneration_strategy_sha256(args)

    reports_dir = Path(args.reports_dir)
    out_root = Path(args.regen_out)
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name in ("accepted", "rejected", "best_failed"):
        (out_root / name).mkdir(parents=True, exist_ok=True)

    attempts_file = _regen_report_path(reports_dir, REGENERATE_ATTEMPTS_NAME, args)
    accepted_file = _regen_report_path(reports_dir, REGENERATE_ACCEPTED_NAME, args)
    still_failed_file = _regen_report_path(reports_dir, REGENERATE_STILL_FAILED_NAME, args)
    summary_file = _regen_report_path(reports_dir, REGENERATE_SUMMARY_NAME, args)
    run_meta_file = _regen_report_path(reports_dir, REGENERATE_RUN_META_NAME, args)
    # Attempts and failures describe this shard invocation and must not leak
    # obsolete queue rows into a later merge. Accepted rows remain append-only
    # so a rerun can safely resume already validated geometries.
    if _safe_text(getattr(args, "run_id", "")):
        if run_meta_file.exists() and not _run_meta_matches(
            run_meta_file, args, int(args.shard_id)
        ):
            raise SystemExit(
                f"Immutable regeneration run metadata mismatch: {run_meta_file}. "
                "Use a new run ID/report directory for a different queue or strategy."
            )
        _ensure_csv(attempts_file, REGEN_ATTEMPT_FIELDS)
        _ensure_csv(accepted_file, REGEN_ACCEPTED_FIELDS)
        _ensure_csv(still_failed_file, REGEN_STILL_FAILED_FIELDS)
        if not run_meta_file.exists():
            _write_regeneration_run_meta(run_meta_file, args)
    else:
        _reset_csv(attempts_file, REGEN_ATTEMPT_FIELDS)
        _ensure_csv(accepted_file, REGEN_ACCEPTED_FIELDS)
        _reset_csv(still_failed_file, REGEN_STILL_FAILED_FIELDS)

    queue = pd.read_csv(queue_path, low_memory=False)
    input_rows = len(queue)
    queue = _regeneration_queue_rows(queue)
    total_fail_rows = len(queue)
    queue["_queue_index"] = range(1, len(queue) + 1)
    if int(args.num_shards) > 1:
        queue = queue[queue.index % int(args.num_shards) == int(args.shard_id)].reset_index(drop=True)
    if args.limit is not None:
        queue = queue.head(int(args.limit)).reset_index(drop=True)

    specs = pd.read_csv(args.specs, low_memory=False)
    specs_by_id = _spec_lookup(specs)
    profiles = _profile_names(args.profile_sequence)
    args.ligtype_override_index = load_ligtype_overrides(args.ligtype_overrides)
    accepted_resume_rows = (
        {} if args.overwrite_accepted else _accepted_report_rows_for_resume(reports_dir, out_root, args)
    )
    current_accepted = _rows_for_regeneration_run(_read_csv_if_exists(accepted_file), args)
    current_accepted_ids = (
        set(current_accepted[gschema.BUILD_ID].astype(str))
        if not current_accepted.empty and gschema.BUILD_ID in current_accepted.columns
        else set()
    )
    skipped_accepted = 0
    prior_attempts = _rows_for_regeneration_run(_read_csv_if_exists(attempts_file), args)
    prior_failed = _rows_for_regeneration_run(_read_csv_if_exists(still_failed_file), args)
    attempts_by_build: dict[str, list[dict]] = {}
    if not prior_attempts.empty and "build_id" in prior_attempts.columns:
        for row in prior_attempts.to_dict("records"):
            attempts_by_build.setdefault(_safe_text(row.get("build_id")), []).append(row)
    terminal_failed_ids = (
        set(prior_failed["build_id"].astype(str))
        if not prior_failed.empty and "build_id" in prior_failed.columns else set()
    )

    attempt_rows: list[dict] = []
    accepted_rows: list[dict] = []
    still_failed_rows: list[dict] = []

    print(
        f"[regenerate-failed] {len(queue)} of {total_fail_rows} regeneration rows "
        f"from {args.regenerate_input} "
        f"[shard {int(args.shard_id)}/{int(args.num_shards)}] "
        f"-> {out_root}",
        flush=True,
    )

    for local_idx, (_, queue_row) in enumerate(queue.iterrows(), 1):
        queue_index = int(queue_row.get("_queue_index", local_idx))
        spec, spec_note = _regeneration_spec(queue_row, specs_by_id)
        if spec is None:
            placeholder_values = {
                "build_id": _safe_text(queue_row.get("build_id", "")),
                "run_id": _safe_text(getattr(args, "run_id", "")),
                "queue_sha256": _safe_text(args.queue_sha256),
                "strategy_sha256": _safe_text(args.strategy_sha256),
            }
            for name in (
                "root_source_build_id", "source_build_id", "parent_build_id",
                "rescue_route", "hypothesis_version",
            ):
                placeholder_values[name] = _safe_text(queue_row.get(name, ""))
            placeholder = pd.Series(placeholder_values)
            still_failed_row = _still_failed_report_row(
                queue_index, placeholder, [], out_root / "best_failed", spec_note,
            )
            still_failed_rows.append(still_failed_row)
            _append_csv_row(still_failed_file, still_failed_row, REGEN_STILL_FAILED_FIELDS)
            print(f"  {local_idx}/{len(queue)} queue#{queue_index} skipped: {spec_note}", flush=True)
            continue

        spec["run_id"] = _safe_text(getattr(args, "run_id", ""))
        spec["queue_sha256"] = _safe_text(args.queue_sha256)
        spec["strategy_sha256"] = _safe_text(args.strategy_sha256)

        build_id = str(spec["build_id"])
        if not args.overwrite_accepted and build_id in accepted_resume_rows:
            candidate_path = Path(_safe_text(
                accepted_resume_rows[build_id].get(gschema.ACCEPTED_XYZ_PATH, "")
            ))
            resumed = _accepted_report_row_from_existing(queue_index, spec, candidate_path, args)
            if resumed is not None:
                skipped_accepted += 1
                # Materialise pilot/canonical acceptance in this shard
                # cardinality only after current dual-QC revalidation.
                if build_id not in current_accepted_ids:
                    _append_csv_row(accepted_file, resumed, REGEN_ACCEPTED_FIELDS)
                    current_accepted_ids.add(build_id)
                print(
                    f"  {local_idx}/{len(queue)} queue#{queue_index} {spec['metal_symbol']:>3} "
                    f"{build_id} skipped: existing accepted dual QC OK",
                    flush=True,
                )
                continue

        if not args.overwrite_accepted:
            existing_xyz = _accepted_xyz_for_spec(spec, out_root)
            if existing_xyz is not None:
                recovered = _accepted_report_row_from_existing(queue_index, spec, existing_xyz, args)
                if recovered is not None:
                    skipped_accepted += 1
                    accepted_resume_rows[build_id] = recovered
                    _append_csv_row(accepted_file, recovered, REGEN_ACCEPTED_FIELDS)
                    current_accepted_ids.add(build_id)
                    print(
                        f"  {local_idx}/{len(queue)} queue#{queue_index} {spec['metal_symbol']:>3} "
                        f"{build_id} skipped: recovered accepted xyz",
                        flush=True,
                    )
                    continue

        if _safe_text(getattr(args, "run_id", "")) and build_id in terminal_failed_ids:
            print(
                f"  {local_idx}/{len(queue)} queue#{queue_index} {spec['metal_symbol']:>3} "
                f"{build_id} skipped: this immutable run already exhausted its strategies",
                flush=True,
            )
            continue

        source_xyz = _source_xyz_path(queue_row, spec)
        row_attempts: list[dict] = list(attempts_by_build.get(build_id, []))
        attempted_numbers = {
            _safe_int(row.get("attempt"), 0) for row in row_attempts
        }
        accepted = None
        for attempt in range(1, max(1, int(args.max_attempts)) + 1):
            if attempt in attempted_numbers:
                continue
            profile = profiles[(attempt - 1) % len(profiles)]
            attempt_row = _run_regeneration_attempt(
                queue_index, spec, source_xyz, attempt, profile, out_root, args,
            )
            row_attempts.append(attempt_row)
            attempts_by_build.setdefault(build_id, []).append(attempt_row)
            attempt_rows.append(attempt_row)
            _append_csv_row(attempts_file, attempt_row, REGEN_ATTEMPT_FIELDS)
            print(
                f"  {local_idx}/{len(queue)} queue#{queue_index} "
                f"{spec['metal_symbol']:>3} {spec['build_id']} "
                f"attempt {attempt}: {attempt_row['attempt_status']} "
                f"{attempt_row.get('qc_class', '')}",
                flush=True,
            )
            if attempt_row.get("accepted_for_clean_3d_features"):
                accepted = attempt_row
                break
            if _repeated_ambiguous_shell(row_attempts, float(args.long_bond_threshold)):
                print(
                    "    stopped: repeated ambiguous shell; prepare an adaptive "
                    "CN/fill hypothesis instead of another seed/profile retry",
                    flush=True,
                )
                break

        if accepted is not None:
            accepted_row = _accepted_report_row(accepted)
            accepted_rows.append(accepted_row)
            accepted_resume_rows[str(accepted_row[gschema.BUILD_ID])] = accepted_row
            _append_csv_row(accepted_file, accepted_row, REGEN_ACCEPTED_FIELDS)
            current_accepted_ids.add(str(accepted_row[gschema.BUILD_ID]))
        else:
            failure_note = (
                row_attempts[-1].get("qc_note") or row_attempts[-1].get("note")
                if row_attempts else "no_attempts"
            )
            still_failed_row = _still_failed_report_row(
                queue_index, spec, row_attempts, out_root / "best_failed", str(failure_note),
            )
            still_failed_rows.append(still_failed_row)
            _append_csv_row(still_failed_file, still_failed_row, REGEN_STILL_FAILED_FIELDS)

    summary_attempts, summary_accepted, summary_still_failed = _read_shard_outputs(
        attempts_file, accepted_file, still_failed_file, args,
    )
    _write_regeneration_summary(
        summary_file,
        args=args,
        input_rows=input_rows,
        queued_rows=len(queue),
        attempt_rows=summary_attempts,
        accepted_rows=summary_accepted,
        still_failed_rows=summary_still_failed,
    )

    print(f"Attempts -> {attempts_file}")
    print(f"Accepted -> {accepted_file}")
    print(f"Still failed -> {still_failed_file}")
    print(f"Summary -> {summary_file}")
    print(f"Skipped already accepted -> {skipped_accepted}")
    return 0


def merge_regenerated(args) -> int:
    reports_dir = Path(args.reports_dir)
    queue_path = Path(args.regenerate_input)
    actual_queue_sha256 = _sha256_file(queue_path)
    requested_queue_sha256 = _safe_text(getattr(args, "queue_sha256", ""))
    if requested_queue_sha256 and requested_queue_sha256 != actual_queue_sha256:
        raise SystemExit(
            "Regeneration queue fingerprint mismatch during merge: "
            f"expected={requested_queue_sha256}, actual={actual_queue_sha256}"
        )
    args.queue_sha256 = actual_queue_sha256
    args.strategy_sha256 = _regeneration_strategy_sha256(args)

    if _safe_text(getattr(args, "run_id", "")):
        meta_paths = _regen_shard_report_paths(
            reports_dir, REGENERATE_RUN_META_NAME, args.num_shards
        )
        missing = [
            str(path) for expected_shard_id, path in enumerate(meta_paths)
            if not _run_meta_matches(path, args, expected_shard_id)
        ]
    else:
        missing = [
            str(path) for path in _regen_shard_report_paths(
                reports_dir, REGENERATE_ATTEMPTS_NAME, args.num_shards
            )
            if not path.exists()
        ] if int(args.num_shards) > 1 else []
    if missing and not args.allow_missing_shard_reports:
        print("Missing regeneration shard attempt reports; refusing partial merge.")
        for path in missing[:20]:
            print(f"  {path}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        print("Use --allow-missing-shard-reports only for manual recovery.")
        return 1

    attempts = _read_regen_shard_reports(reports_dir, REGENERATE_ATTEMPTS_NAME, args.num_shards)
    accepted = _read_regen_shard_reports(reports_dir, REGENERATE_ACCEPTED_NAME, args.num_shards)
    attempts = _rows_for_regeneration_run(attempts, args)
    accepted = _rows_for_regeneration_run(accepted, args)
    # Accepted geometries are durable across pilot/full cardinality changes.
    # Include canonical and all prior shard layouts; current-queue filtering
    # and build-id deduplication below prevent stale rows from leaking in.
    durable_accepted = [
        _read_csv_if_exists(reports_dir / REGENERATE_ACCEPTED_NAME),
        _read_all_regen_shard_reports(reports_dir, REGENERATE_ACCEPTED_NAME),
    ]
    accepted = pd.concat(
        [frame for frame in durable_accepted if not frame.empty],
        ignore_index=True, sort=False,
    ) if any(not frame.empty for frame in durable_accepted) else pd.DataFrame()
    accepted = _rows_for_regeneration_run(accepted, args)
    still_failed = _read_regen_shard_reports(reports_dir, REGENERATE_STILL_FAILED_NAME, args.num_shards)
    still_failed = _rows_for_regeneration_run(still_failed, args)
    if attempts.empty and accepted.empty and still_failed.empty:
        print(
            f"No regeneration shard reports found in {reports_dir} "
            f"for num_shards={args.num_shards}."
        )
        return 1

    if not attempts.empty:
        attempts = attempts.sort_values(
            [c for c in ["queue_index", "attempt"] if c in attempts.columns],
            kind="stable",
        )
    if not accepted.empty and "queue_index" in accepted.columns:
        accepted = accepted.sort_values("queue_index", kind="stable")
    if not still_failed.empty and "queue_index" in still_failed.columns:
        still_failed = still_failed.sort_values("queue_index", kind="stable")
    if not accepted.empty and not still_failed.empty and gschema.BUILD_ID in accepted.columns and "build_id" in still_failed.columns:
        ok_mask = (
            accepted[gschema.QC_CLASS].astype(str) == "OK"
            if gschema.QC_CLASS in accepted.columns
            else pd.Series(False, index=accepted.index)
        )
        accepted_ok = accepted[ok_mask]
        accepted_build_ids = set(accepted_ok[gschema.BUILD_ID].astype(str))
        still_failed = still_failed[~still_failed["build_id"].astype(str).isin(accepted_build_ids)]

    attempts_file = reports_dir / REGENERATE_ATTEMPTS_NAME
    accepted_file = reports_dir / REGENERATE_ACCEPTED_NAME
    still_failed_file = reports_dir / REGENERATE_STILL_FAILED_NAME
    summary_file = reports_dir / REGENERATE_SUMMARY_NAME

    input_rows = 0
    queued_rows = 0
    if Path(args.regenerate_input).exists():
        queue = pd.read_csv(args.regenerate_input, low_memory=False)
        input_rows = len(queue)
        queue = _regeneration_queue_rows(queue)
        queued_rows = len(queue)
        accepted = _dual_qc_accepted_for_queue(accepted, queue, args)
        attempts, accepted, still_failed = _reports_for_current_regeneration_queue(
            queue, attempts, accepted, still_failed,
        )

    # Keep zero-row merged reports parseable.  This also upgrades historical
    # one-byte/headerless files when the merge is rerun.
    if attempts.empty and len(attempts.columns) == 0:
        attempts = pd.DataFrame(columns=REGEN_ATTEMPT_FIELDS)
    if accepted.empty and len(accepted.columns) == 0:
        accepted = pd.DataFrame(columns=REGEN_ACCEPTED_FIELDS)
    if still_failed.empty and len(still_failed.columns) == 0:
        still_failed = pd.DataFrame(columns=REGEN_STILL_FAILED_FIELDS)

    _write_csv_atomic(attempts, attempts_file)
    _write_csv_atomic(accepted, accepted_file)
    _write_csv_atomic(still_failed, still_failed_file)

    progress = _write_regeneration_summary(
        summary_file,
        args=args,
        input_rows=input_rows,
        queued_rows=queued_rows,
        attempt_rows=attempts.to_dict("records"),
        accepted_rows=accepted.to_dict("records"),
        still_failed_rows=still_failed.to_dict("records"),
        missing_shard_reports=len(missing),
    )

    print(f"Merged shard attempts -> {attempts_file} ({len(attempts)} rows)")
    print(f"Merged shard accepted -> {accepted_file} ({len(accepted)} rows)")
    print(f"Merged shard still failed -> {still_failed_file} ({len(still_failed)} rows)")
    print(f"Merged summary -> {summary_file}")
    if progress["incomplete_rows"] or progress["missing_shard_reports"]:
        print(
            "WARNING: partial regeneration merge: "
            f"completed={progress['completed_rows']}/{queued_rows}, "
            f"missing_shard_reports={progress['missing_shard_reports']}. "
            "Rerun submit-regeneration to resume accepted rows and retry the remainder."
        )
        return 2
    return 0


# ---------------------------------------------------------------------------
# Modes: run shard, recover, merge, audit
# ---------------------------------------------------------------------------
def _resolve_retry_statuses(args) -> set[str]:
    retry: set[str] = set()
    if args.retry_failed:
        retry |= {s for s in RETRYABLE_STATUSES if s.startswith("failed_")}
    for s in args.retry_status or []:
        if s not in RETRYABLE_STATUSES:
            raise SystemExit(f"--retry-status {s!r} not retryable. Choose from {sorted(RETRYABLE_STATUSES)}")
        retry.add(s)
    return retry


def _iter_recorded_frames(num_shards: int = 16, index_tag: str = ""):
    for path in shard_index_parts(num_shards, index_tag):
        try:
            yield read_index_csv(path)
        except Exception:
            continue


def _all_recorded_status(num_shards: int = 16, index_tag: str = "") -> dict[str, str]:
    recorded: dict[str, str] = {}
    for df in _iter_recorded_frames(num_shards, index_tag):
        if "build_id" not in df.columns or "status" not in df.columns:
            continue
        for bid, status in zip(df["build_id"].astype(str), df["status"].astype(str)):
            if bid not in recorded or (status in KEEP_STATUSES and recorded[bid] not in KEEP_STATUSES):
                recorded[bid] = status
    return recorded


def _recorded_build_ids_with_status(statuses: set[str], num_shards: int = 16,
                                    index_tag: str = "") -> set[str]:
    build_ids: set[str] = set()
    for df in _iter_recorded_frames(num_shards, index_tag):
        if "build_id" not in df.columns or "status" not in df.columns:
            continue
        mask = df["status"].astype(str).isin(statuses)
        build_ids |= set(df.loc[mask, "build_id"].astype(str))
    return build_ids


def _force_rebuild_build_ids(
    recorded: dict[str, str],
    retry_statuses: set[str],
    legacy_simplified_ids: set[str],
) -> set[str]:
    return {
        bid for bid, status in recorded.items()
        if status in retry_statuses
        and (status in FORCE_REBUILD_STATUSES or bid in legacy_simplified_ids)
    }


def _planned_specs(args) -> pd.DataFrame:
    specs = pd.read_csv(args.specs, low_memory=False)
    if "geometry_status" in specs.columns:
        specs = specs[specs["geometry_status"] == "planned"]
    return specs.reset_index(drop=True)


def _missing_geometry_specs(specs: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    """Return planned specs whose canonical XYZ is absent or invalid on disk."""
    missing = [
        not valid_xyz(expected_paths(spec, out_root)[0])
        for _, spec in specs.iterrows()
    ]
    return specs.loc[missing].reset_index(drop=True)


def count_missing_geometries(args) -> int:
    missing = _missing_geometry_specs(_planned_specs(args), Path(args.out))
    print(len(missing))
    return 0


def plan_missing_geometries(args) -> int:
    """Validate the exact disk-missing queue, ligTypes, and shard allocation."""
    missing = _missing_geometry_specs(_planned_specs(args), Path(args.out))
    expected = getattr(args, "expected_missing_count", None)
    if expected is not None and len(missing) != int(expected):
        raise SystemExit(
            f"Expected exactly {expected} missing canonical XYZ files, found {len(missing)}"
        )
    if len(missing) != int(args.num_shards):
        raise SystemExit(
            f"One-molecule-per-shard invariant requires num_shards={len(missing)}, "
            f"got {args.num_shards}"
        )

    args.ligtype_override_index = load_ligtype_overrides(args.ligtype_overrides)
    rows = []
    for queue_index, (_, spec) in enumerate(missing.iterrows()):
        ligtype = _ligtype_override_for_spec(spec, args)
        if not ligtype:
            raise SystemExit(
                f"Missing explicit ligType override for build_id={spec['build_id']}"
            )
        rows.append({
            "shard": queue_index % int(args.num_shards),
            "build_id": str(spec["build_id"]),
            "metal": str(spec["metal_symbol"]),
            "denticity": _safe_int(spec.get("DENTATE")),
            "ligType": ligtype,
        })

    plan = pd.DataFrame(rows)
    counts = plan["shard"].value_counts().to_dict() if not plan.empty else {}
    if len(counts) != int(args.num_shards) or any(int(v) != 1 for v in counts.values()):
        raise SystemExit(f"Invalid missing-geometry shard allocation: {counts}")
    queue_output = getattr(args, "missing_queue_output", None)
    if queue_output:
        queue_output = Path(queue_output)
        queue_output.parent.mkdir(parents=True, exist_ok=True)
        tmp_output = queue_output.with_suffix(queue_output.suffix + ".tmp")
        missing.to_csv(tmp_output, index=False)
        os.replace(tmp_output, queue_output)
    print(plan.to_csv(index=False).rstrip())
    print(
        f"PREFLIGHT_OK missing={len(missing)} shards={args.num_shards} "
        "per_shard=1 explicit_ligtype=yes"
    )
    if queue_output:
        print(f"FROZEN_QUEUE={queue_output}")
    return 0


def _assigned_specs(specs: pd.DataFrame, num_shards: int, shard_id: int) -> pd.DataFrame:
    return specs[specs.index % num_shards == shard_id].reset_index(drop=True)


def run_shard(args) -> int:
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must be in [0, num-shards-1]")

    retry_statuses = _resolve_retry_statuses(args)
    specs = _planned_specs(args)
    if args.missing_only and not getattr(args, "fixed_missing_queue", False):
        # Filter first, then shard.  With 23 missing rows and 23 shards this is
        # the guarantee that every task receives exactly one missing molecule.
        specs = _missing_geometry_specs(specs, Path(args.out))
    mine = _assigned_specs(specs, args.num_shards, args.shard_id)
    if args.limit:
        mine = mine.head(args.limit)

    index_tag = _validated_index_tag(args.index_tag)
    shard_index = PROCESSED_DIR / _shard_artifact_name(
        "geometry_index", args.shard_id, args.num_shards, index_tag
    )
    recorded = _all_recorded_status(args.num_shards, index_tag) if args.skip_existing else {}
    legacy_simplified_ids = (
        _recorded_build_ids_with_status(FORCE_REBUILD_STATUSES, args.num_shards, index_tag)
        if args.skip_existing else set()
    )
    args.force_rebuild_build_ids = _force_rebuild_build_ids(
        recorded, retry_statuses, legacy_simplified_ids
    )
    args.ligtype_override_index = load_ligtype_overrides(args.ligtype_overrides)

    def wanted(spec: pd.Series) -> bool:
        if args.missing_only:
            # Disk state is the source of truth here. A failed/skipped row is
            # still eligible when its canonical XYZ is absent, while any valid
            # existing geometry remains untouched regardless of stale indexes.
            return not valid_xyz(expected_paths(spec, Path(args.out))[0])
        build_id = str(spec["build_id"])
        status = recorded.get(build_id)
        if status is None:
            return not args.retry_only
        if status in retry_statuses:
            return True
        if status in KEEP_STATUSES:
            return False
        return False

    # Write the ligand pre-screen audit (heavy atoms / MW / rotatable bonds).
    prescreen_path = (
        AUDIT_DIR / _shard_artifact_name(
            "ligand_prescreen", args.shard_id, args.num_shards, index_tag
        )
        if args.num_shards > 1 else PRESCREEN_FILE
    )
    _write_prescreen(mine, prescreen_path)

    known_bad = load_known_bad()
    todo = [spec for _, spec in mine.iterrows() if wanted(spec)]
    mode = ("missing-only" if args.missing_only else
            "retry-only" if args.retry_only else "fill+retry")
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(mine)} assigned, {len(todo)} to process "
          f"[{mode}; retry={sorted(retry_statuses) or 'none'}] -> {shard_index.name}", flush=True)

    counts: dict[str, int] = {}
    for i, spec in enumerate(todo, 1):
        try:
            row = assemble_one(spec, args, known_bad)
        except Exception as exc:
            traceback.print_exc()
            row = index_row(spec, status="failed_exception",
                            note=f"row_controller_exception:{type(exc).__name__}: {str(exc)[:240]}")
        append_index_row(shard_index, row)   # IMMEDIATE append
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        print(f"  {i}/{len(todo)}  {spec['metal_symbol']:>3}  {row['status']:<28} {str(row['note'])[:60]}",
              flush=True)

    print(f"[shard {args.shard_id}] done. status counts: {json.dumps(counts)}", flush=True)
    merge_hint = (
        "python scripts/build_unique_geometries.py --merge-index-only "
        f"--num-shards {args.num_shards}"
    )
    if index_tag:
        merge_hint += f" --index-tag {index_tag}"
    print(f"Combine with: {merge_hint}")
    return 0


def _write_prescreen(specs: pd.DataFrame, path: Path = PRESCREEN_FILE) -> None:
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        _prescreen_frame(specs).to_csv(path, index=False)
    except Exception:
        pass


def _safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fill_heavy_atoms(fill_ligand: str) -> int:
    fill = str(fill_ligand or "water").lower()
    if fill == "nitrate":
        return 4
    return 1


def _prescreen_frame(specs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, spec in specs.iterrows():
        m = ligand_metrics(str(spec["SMILES_FOR_ARCHITECTOR"]))
        heavy_atoms = _safe_int(m.get("heavy_atoms", 0))
        n_ligs = _safe_int(spec.get("n_ligs", 0))
        n_fill = _safe_int(spec.get("n_fill", 0))
        fill_ligand = str(spec.get("fill_ligand", spec.get("inner_sphere_anion", "water")))
        heavy_ligand = int(heavy_atoms >= HEAVY_LIGAND_ATOM_THRESHOLD)
        rows.append({
            "build_id": spec["build_id"],
            "metal_symbol": spec["metal_symbol"],
            "coreCN": spec.get("coreCN", ""),
            "DENTATE": spec.get("DENTATE", ""),
            "n_ligs": spec.get("n_ligs", ""),
            "inner_sphere_anion": spec.get("inner_sphere_anion", ""),
            "fill_ligand": fill_ligand,
            "n_fill": spec.get("n_fill", ""),
            "rows_covered": spec.get("rows_covered", ""),
            "heavy_atoms": heavy_atoms if m else "",
            "mol_weight": m.get("mol_weight", ""),
            "n_rotatable_bonds": m.get("n_rotatable_bonds", ""),
            "estimated_complex_heavy_atoms": (
                heavy_atoms * n_ligs + n_fill * _fill_heavy_atoms(fill_ligand)
                if m else ""
            ),
            "heavy_ligand": heavy_ligand,
            "recommended_profile_sequence": (
                HARD_LIGAND_PROFILE_SEQUENCE if heavy_ligand else DEFAULT_PROFILE_SEQUENCE
            ),
            "SMILES_FOR_ARCHITECTOR": spec["SMILES_FOR_ARCHITECTOR"],
        })
    return pd.DataFrame(rows)


def prescreen_only(args) -> int:
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must be in [0, num-shards-1]")
    specs = _planned_specs(args)
    mine = specs[specs.index % args.num_shards == args.shard_id].reset_index(drop=True)
    if args.limit:
        mine = mine.head(args.limit)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    frame = _prescreen_frame(mine)
    frame.to_csv(PRESCREEN_FILE, index=False)
    heavy = frame[frame["heavy_ligand"].astype(str).isin({"1", "True", "true"})]
    summary = {
        "specs_screened": int(len(frame)),
        "heavy_ligand_threshold_heavy_atoms": HEAVY_LIGAND_ATOM_THRESHOLD,
        "heavy_ligand_specs": int(len(heavy)),
        "heavy_ligand_rows_covered": int(pd.to_numeric(
            heavy.get("rows_covered", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0).sum()),
        "max_heavy_atoms": int(pd.to_numeric(frame.get("heavy_atoms"), errors="coerce").max())
        if len(frame) else 0,
        "max_rotatable_bonds": int(pd.to_numeric(frame.get("n_rotatable_bonds"), errors="coerce").max())
        if len(frame) else 0,
        "recommended_hard_ligand_command": (
            "python scripts/build_unique_geometries.py --hard-ligand-mode "
            "--retry-status failed_ligtype --retry-status failed_no_structures"
        ),
    }
    PRESCREEN_SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Prescreen table -> {PRESCREEN_FILE}")
    print(f"Prescreen summary -> {PRESCREEN_SUMMARY_FILE}")
    return 0


def recover_index(args) -> int:
    specs = pd.read_csv(args.specs, low_memory=False)
    by_id = {str(r["build_id"]): r for _, r in specs.iterrows()}
    xyz_files = sorted(Path(args.out).rglob("*.xyz"))
    rows, unmatched = [], 0
    for xyz in xyz_files:
        if xyz.name.endswith(".tmp.xyz"):
            continue
        build_id = xyz.stem.rsplit("_", 1)[-1]
        spec = by_id.get(build_id)
        if spec is None:
            unmatched += 1
            continue
        mol2 = xyz.with_suffix(".mol2")
        if valid_xyz(xyz):
            qc_status, _ = qc_xyz(xyz, spec)
            status = "existing_ok" if qc_status == "accepted" else "failed_qc"
        else:
            qc_status, status = "QC_FAILED", "failed_invalid_xyz"
        rows.append(index_row(spec, status=status, note="recovered_from_disk", qc_status=qc_status,
                              xyz_path=str(xyz) if valid_xyz(xyz) else "",
                              mol2_path=str(mol2) if mol2.exists() else ""))
    pd.DataFrame(rows, columns=INDEX_FIELDS).to_csv(RECOVERED_INDEX_FILE, index=False)
    print(f"Recovered {len(rows)} index rows from {len(xyz_files)} xyz files "
          f"({unmatched} unmatched) -> {RECOVERED_INDEX_FILE}")
    print("Now run: python scripts/build_unique_geometries.py --merge-index-only")
    return 0


def _dedup_best(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    legacy_simplified_ids = set(
        combined.loc[
            combined["status"].astype(str) == "ok_simplified_ligand",
            "build_id",
        ].astype(str)
    )
    if legacy_simplified_ids:
        status = combined["status"].astype(str)
        build_ids = combined["build_id"].astype(str)
        recovered_mask = build_ids.isin(legacy_simplified_ids) & status.eq("existing_ok")
        combined.loc[recovered_mask, "status"] = "ok_simplified_ligand"
        combined.loc[recovered_mask, "simplified_ligand"] = True
        combined.loc[recovered_mask, "note"] = (
            combined.loc[recovered_mask, "note"].fillna("").astype(str)
            + "; recovered_existing_matches_legacy_simplified_ligand"
        ).str.strip("; ")
    combined["_source_order"] = range(len(combined))
    has_xyz = combined["xyz_path"].fillna("").astype(str).str.strip().ne("")
    status = combined["status"].astype(str)
    combined["_rank"] = 9
    combined.loc[has_xyz, "_rank"] = 3
    combined.loc[has_xyz & status.eq("failed_qc"), "_rank"] = 2
    combined.loc[status.eq("existing_ok"), "_rank"] = 1
    combined.loc[status.eq("ok"), "_rank"] = 0
    return (combined.sort_values(["_rank", "_source_order"], kind="mergesort")
            .drop_duplicates(subset=["build_id"], keep="first")
            .drop(columns=["_rank", "_source_order"]))


def merge_indices(args) -> int:
    index_tag = _validated_index_tag(args.index_tag)
    parts = shard_index_parts(args.num_shards, index_tag)
    extra = [] if index_tag else (
        [RECOVERED_INDEX_FILE] if RECOVERED_INDEX_FILE.exists() else []
    )
    if not parts and not extra:
        tag_hint = f"_{index_tag}" if index_tag else ""
        print(f"No geometry_index{tag_hint}_shard*of{args.num_shards}.csv files found.")
        return 1
    frames = [read_index_csv(p) for p in (*parts, *extra)]
    combined = _dedup_best(frames)
    final_file = (
        PROCESSED_DIR / f"geometry_index_{index_tag}_merged.csv"
        if index_tag else FINAL_MERGED_INDEX_FILE
    )
    merge_file = (
        PROCESSED_DIR / f"geometry_index_{index_tag}_for_merge.csv"
        if index_tag else MERGE_INDEX_FILE
    )
    unsuccessful_file = (
        PROCESSED_DIR / f"geometry_index_{index_tag}_unsuccessful.csv"
        if index_tag else UNSUCCESSFUL_INDEX_FILE
    )
    combined.to_csv(final_file, index=False)
    combined.to_csv(merge_file, index=False)

    success = combined["status"].isin(SUCCESS_STATUSES)
    combined[~success].to_csv(unsuccessful_file, index=False)

    print(f"Merged {len(parts)} shard index files (+{len(extra)} recovered) "
          f"-> {final_file} ({len(combined)} complexes, {int(success.sum())} with geometry)")
    print(f"Unsuccessful -> {unsuccessful_file} ({int((~success).sum())} rows)")
    print("Status breakdown:", json.dumps(combined["status"].value_counts().to_dict()))
    return 0


def audit_xyz(args) -> int:
    merged_file = current_merged_index_file()
    if not merged_file.exists():
        print(f"No merge index at {merged_file}; run --merge-index-only first.")
        return 1
    idx = pd.read_csv(merged_file, low_memory=False)
    disk = {str(p.resolve()) for p in Path(args.out).rglob("*.xyz") if not p.name.endswith(".tmp.xyz")}

    indexed = idx[idx["xyz_path"].notna() & (idx["xyz_path"].astype(str) != "")]
    resolved = {str(Path(p).resolve()): p for p in indexed["xyz_path"].astype(str)}

    unindexed = sorted(disk - set(resolved))
    missing = sorted(p for rp, p in resolved.items() if rp not in disk)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"xyz_path": unindexed}).to_csv(AUDIT_DIR / "xyz_audit_unindexed_files.csv", index=False)
    pd.DataFrame({"xyz_path": missing}).to_csv(AUDIT_DIR / "xyz_audit_missing_files.csv", index=False)

    print(f"xyz on disk:          {len(disk)}")
    print(f"xyz in index:         {len(resolved)}")
    print(f"disk not indexed:     {len(unindexed)}  -> xyz_audit_unindexed_files.csv")
    print(f"index missing on disk:{len(missing)}  -> xyz_audit_missing_files.csv")
    return 0


def triage_nearest_corecn(args) -> int:
    merged_file = current_merged_index_file()
    if not merged_file.exists():
        print(f"No merge index at {merged_file}; run --merge-index-only first.")
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(merged_file, low_memory=False)
    if "status" not in idx.columns or "xyz_path" not in idx.columns:
        print(f"{merged_file} must contain status and xyz_path columns.")
        return 1

    success = idx[
        idx["status"].astype(str).isin(SUCCESS_STATUSES)
        & idx["xyz_path"].notna()
        & (idx["xyz_path"].astype(str) != "")
    ].copy()

    rows = []
    for _, row in success.iterrows():
        out = row.to_dict()
        xyz_path = Path(str(row.get("xyz_path", "")))
        out["xyz_exists"] = xyz_path.exists()
        if out["xyz_exists"]:
            qc = nearest_corecn_qc_xyz(
                xyz_path,
                row,
                long_bond_threshold=float(args.long_bond_threshold),
                borderline_longish_threshold=float(args.borderline_longish_threshold),
                ambiguous_gap_threshold=float(args.ambiguous_gap_threshold),
            )
        else:
            qc = {
                "qc_class": "QC_FAILED",
                "qc_note": "indexed_xyz_path_missing_on_disk",
                "nearest_coreCN_sig": "",
                "coreCN_max_dist": "",
                "next_donor_dist": "",
                "gap_after_coreCN": "",
                "all_nearest": "",
            }
        out.update(qc)
        rows.append(out)

    frame = pd.DataFrame(rows)
    if frame.empty:
        print(f"No successful geometries with xyz_path found in {merged_file}.")
        return 1

    ok = frame[frame["qc_class"].astype(str) == "OK"].copy()
    fail_long = frame[frame["qc_class"].astype(str) == "FAIL_LONG_BOND"].copy()
    borderline = frame[
        frame["qc_class"].astype(str).isin(
            {"BORDERLINE_AMBIGUOUS_SHELL", "BORDERLINE_LONGISH"}
        )
    ].copy()

    ok_file = REPORTS_DIR / "geometry_ok_for_features.csv"
    fail_file = REPORTS_DIR / "geometry_regenerate_fail_long_bond.csv"
    borderline_file = REPORTS_DIR / "geometry_borderline_review.csv"
    summary_file = REPORTS_DIR / "nearest_corecn_geometry_qc_summary.txt"

    ok.to_csv(ok_file, index=False)
    fail_long.to_csv(fail_file, index=False)
    borderline.to_csv(borderline_file, index=False)

    counts = {str(k): int(v) for k, v in frame["qc_class"].value_counts().items()}
    summary = [
        "Nearest-coreCN geometry QC summary",
        "",
        f"Input merged index: {merged_file}",
        f"Rows checked: {len(frame)}",
        "QC class distribution:",
        json.dumps(counts, indent=2, sort_keys=True),
        "",
        f"OK for 3D features: {len(ok)} -> {ok_file}",
        f"Regenerate FAIL_LONG_BOND: {len(fail_long)} -> {fail_file}",
        f"Borderline review: {len(borderline)} -> {borderline_file}",
        "",
        "Thresholds:",
        f"  FAIL_LONG_BOND if coreCN_max_dist > {float(args.long_bond_threshold):.2f} A",
        f"  BORDERLINE_AMBIGUOUS_SHELL if gap_after_coreCN < {float(args.ambiguous_gap_threshold):.2f} A",
        f"  BORDERLINE_LONGISH if coreCN_max_dist > {float(args.borderline_longish_threshold):.2f} A",
    ]
    summary_file.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Rows checked: {len(frame)}")
    print("QC class distribution:", json.dumps(counts, sort_keys=True))
    print(f"OK -> {ok_file} ({len(ok)} rows)")
    print(f"Regenerate -> {fail_file} ({len(fail_long)} rows)")
    print(f"Borderline -> {borderline_file} ({len(borderline)} rows)")
    print(f"Summary -> {summary_file}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=[
        "count-missing-geometries", "plan-missing-geometries", "regenerate-failed",
        "prepare-missing-regeneration", "prepare-family-regeneration",
        "prepare-adaptive-regeneration",
        "family-regeneration-status", "prepare-all-remaining",
        "all-remaining-status",
        "prepare-hard10-rescue", "plan-regeneration-shards", "merge-regenerated",
        "triage-nearest-corecn",
    ],
                    help="Run a focused mode, e.g. regenerate-failed or merge-regenerated.")
    ap.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    ap.add_argument("--out", type=Path, default=GEOMETRY_DIR)
    ap.add_argument("--input", dest="regenerate_input", type=Path,
                    default=DEFAULT_REGENERATE_FAIL_LONG_BOND_INPUT,
                    help="FAIL_LONG_BOND queue CSV for regenerate-failed mode.")
    ap.add_argument("--still-failed-input", type=Path,
                    default=DEFAULT_REGENERATE_STILL_FAILED_INPUT,
                    help="Merged still-failed report used to select rows with no generated XYZ.")
    ap.add_argument("--rescue-queue-output", type=Path,
                    default=DEFAULT_MISSING_REGENERATION_QUEUE,
                    help="Focused no-XYZ queue written by prepare-missing-regeneration.")
    ap.add_argument("--hard10-no-structures-queue-output", type=Path,
                    default=DEFAULT_HARD10_NO_STRUCTURES_QUEUE,
                    help="Five-row failed_no_structures rescue queue output.")
    ap.add_argument("--hard10-native-crash-queue-output", type=Path,
                    default=DEFAULT_HARD10_NATIVE_CRASH_QUEUE,
                    help="Five-row native-crash rescue queue output.")
    ap.add_argument("--family-plan-dir", type=Path,
                    default=DEFAULT_FAMILY_REGENERATION_DIR,
                    help="Output directory for chemistry-family regeneration queues.")
    ap.add_argument("--family-runs-dir", type=Path,
                    default=REPORTS_DIR / "family_regeneration_runs",
                    help="Root directory containing per-family regeneration reports.")
    ap.add_argument("--adaptive-input", type=Path,
                    default=DEFAULT_ADAPTIVE_REGENERATION_INPUT,
                    help="Still-failed report used to diagnose adjacent-CN hypotheses.")
    ap.add_argument("--adaptive-output", type=Path,
                    default=DEFAULT_ADAPTIVE_REGENERATION_QUEUE,
                    help="Explicit adaptive CN/fill regeneration queue output.")
    ap.add_argument("--adaptive-attempts-input", type=Path, default=None,
                    help="Attempt-level report used to prove repeated ambiguous-shell diagnoses.")
    ap.add_argument("--all-remaining-plan-dir", type=Path,
                    default=DEFAULT_ALL_REMAINING_PLAN_DIR,
                    help="Immutable plan, queues, manifest, and run configuration output.")
    ap.add_argument("--all-remaining-runs-dir", type=Path,
                    default=DEFAULT_ALL_REMAINING_RUNS_DIR,
                    help="Per-route reports and final all-remaining status root.")
    ap.add_argument("--all-remaining-out-dir", type=Path,
                    default=DEFAULT_ALL_REMAINING_OUT_DIR,
                    help="Isolated geometry output root for all-remaining regeneration.")
    ap.add_argument("--geometry-index", type=Path, default=FINAL_MERGED_INDEX_FILE,
                    help="Canonical merged geometry index audited by prepare-all-remaining.")
    ap.add_argument("--historical-reports-dir", type=Path, default=REPORTS_DIR,
                    help="Root containing historical accepted regeneration reports.")
    ap.add_argument("--hypothesis-version", default=ALL_REMAINING_VERSION,
                    help="Version token included in new hypothesis IDs and immutable run IDs.")
    ap.add_argument(
        "--remaining-scope",
        choices=("known-unfinished", "strict-baseline-audit"),
        default="known-unfinished",
        help=(
            "Default: regenerate only known unfinished sources. "
            "strict-baseline-audit explicitly revalidates every historical success."
        ),
    )
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR,
                    help="Directory for regenerate-failed report CSV/TXT outputs.")
    ap.add_argument("--regen-out", type=Path, default=REGENERATED_FAIL_LONG_BOND_DIR,
                    help="Separate output root for regenerated geometries.")
    ap.add_argument("--run-id", default="",
                    help="Immutable regeneration invocation ID; enables strict resume semantics.")
    ap.add_argument("--queue-sha256", default="",
                    help="Expected SHA-256 of the frozen regeneration queue.")
    ap.add_argument("--ligtype-overrides", type=Path, default=LIGTYPE_OVERRIDES_FILE,
                    help="Manual ligType override CSV for failed_ligtype rescue.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-symmetries", type=int, default=40)
    ap.add_argument("--n-symmetries-step", type=int, default=10,
                    help="Increase n_symmetries by this amount per regeneration attempt.")
    ap.add_argument("--n-conformers", type=int, default=5)
    ap.add_argument("--n-conformers-step", type=int, default=1,
                    help="Increase n_conformers by this amount per regeneration attempt.")
    ap.add_argument("--xtb-max-iterations", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0xF00D)
    ap.add_argument("--seed-step", type=int, default=7919,
                    help="Per-attempt seed offset for build and regenerate modes.")
    ap.add_argument("--timeout-per-complex", type=int, default=1800,
                    help="Per-complex wall-clock limit in seconds (default 1800).")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="Max tries for non-deterministic failures (native crash / no_structures).")
    ap.add_argument("--max-total-seconds", type=int, default=0,
                    help="Stop launching child attempts after this per-molecule budget (0 disables).")
    ap.add_argument("--expected-missing-count", type=int, default=None,
                    help="plan-missing-geometries: fail unless exactly this many XYZ files are missing.")
    ap.add_argument("--missing-queue-output", type=Path, default=None,
                    help="plan-missing-geometries: atomically write the frozen missing-spec queue here.")
    ap.add_argument("--limit", type=int, default=None, help="Assemble at most N specs (smoke test).")
    ap.add_argument("--overwrite-accepted", action="store_true",
                    help="Regenerate even when this build_id already has accepted qc_class OK output.")
    ap.add_argument("--allow-missing-shard-reports", action="store_true",
                    help="merge-regenerated: allow manual partial merges when some shard reports are missing.")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                    help="Skip specs whose valid .xyz already exists (default).")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="Recompute even if a valid .xyz already exists.")
    ap.add_argument("--profile", choices=sorted(BUILD_PROFILES), default="standard", help=argparse.SUPPRESS)
    ap.add_argument("--profile-sequence", default=DEFAULT_PROFILE_SEQUENCE,
                    help=f"Comma-separated build profiles. Default: {DEFAULT_PROFILE_SEQUENCE}. "
                         f"Hard-ligand: {HARD_LIGAND_PROFILE_SEQUENCE}.")
    ap.add_argument("--auto-hard-ligand", action="store_true", default=True,
                    help="Auto-escalate bulky ligands to the hard-ligand profile sequence (default).")
    ap.add_argument("--no-auto-hard-ligand", dest="auto_hard_ligand", action="store_false")
    ap.add_argument("--allow-simplified-ligand", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--ignore-known-bad", action="store_true",
                    help="Retry ligands in known_bad_ligtype.csv instead of skipping them.")
    ap.add_argument("--hard-ligand-mode", action="store_true",
                    help="Hard-ligand fallback sequence + bypass known-bad cache.")
    ap.add_argument("--fixed-coordlist", action="store_true", default=True,
                    help="Regeneration mode: preserve the original COORDLIST (enforced).")
    ap.add_argument("--rerun-qc", action="store_true", default=True,
                    help="Regeneration mode: rerun nearest-coreCN QC after every attempt (enforced).")
    ap.add_argument("--long-bond-threshold", type=float, default=DEFAULT_LONG_BOND_THRESHOLD,
                    help="Nearest-coreCN FAIL_LONG_BOND cutoff in angstrom.")
    ap.add_argument("--borderline-longish-threshold", type=float,
                    default=DEFAULT_BORDERLINE_LONGISH_THRESHOLD,
                    help="Nearest-coreCN BORDERLINE_LONGISH cutoff in angstrom.")
    ap.add_argument("--ambiguous-gap-threshold", type=float, default=DEFAULT_AMBIGUOUS_GAP_THRESHOLD,
                    help="Nearest-coreCN ambiguous shell gap cutoff in angstrom.")
    ap.add_argument("--retry-status", action="append", metavar="STATUS",
                    help="Re-attempt specs previously recorded with this status (repeatable).")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Re-attempt every spec carrying any failed_* status.")
    ap.add_argument("--missing-only", action="store_true",
                    help="Only process specs whose canonical XYZ is absent or invalid on disk.")
    ap.add_argument("--fixed-missing-queue", action="store_true",
                    help="--specs is a frozen missing queue; shard it before disk-state skip checks.")
    ap.add_argument("--index-tag", default="",
                    help="Isolate shard indexes/audits under a safe filename tag.")
    ap.add_argument("--retry-only", action="store_true",
                    help="Only re-attempt retry-selected statuses; skip missing specs.")
    # modes
    ap.add_argument("--build-one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--build-id", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--spec-json", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--override-smiles", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--override-coordlist", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--override-ligtype", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--recover-index-only", action="store_true",
                    help="Rebuild index rows from existing .xyz files, then exit.")
    ap.add_argument("--merge-index-only", action="store_true",
                    help="Combine shard indices into merged + unsuccessful indices, then exit.")
    ap.add_argument("--audit-xyz", action="store_true",
                    help="Reconcile .xyz files on disk vs the merge index, then exit.")
    ap.add_argument("--prescreen-only", action="store_true",
                    help="Write ligand size / hard-ligand audit without running Architector.")
    args = ap.parse_args()

    if args.hard_ligand_mode:
        args.profile_sequence = HARD_LIGAND_PROFILE_SEQUENCE
        args.ignore_known_bad = True

    if args.allow_simplified_ligand:
        raise SystemExit(
            "--allow-simplified-ligand is disabled: generated geometries must use "
            "the original SMILES_FOR_ARCHITECTOR."
        )

    if args.build_one:
        return _build_one_child(args)
    if args.command == "regenerate-failed":
        return regenerate_failed(args)
    if args.command == "count-missing-geometries":
        return count_missing_geometries(args)
    if args.command == "plan-missing-geometries":
        return plan_missing_geometries(args)
    if args.command == "prepare-missing-regeneration":
        return prepare_missing_regeneration(args)
    if args.command == "prepare-family-regeneration":
        return prepare_family_regeneration(args)
    if args.command == "prepare-adaptive-regeneration":
        return prepare_adaptive_regeneration(args)
    if args.command == "family-regeneration-status":
        return family_regeneration_status(args)
    if args.command == "prepare-all-remaining":
        return prepare_all_remaining(args)
    if args.command == "all-remaining-status":
        return all_remaining_status(args)
    if args.command == "prepare-hard10-rescue":
        return prepare_hard10_rescue(args)
    if args.command == "plan-regeneration-shards":
        return plan_regeneration_shards(args)
    if args.command == "merge-regenerated":
        return merge_regenerated(args)
    if args.command == "triage-nearest-corecn":
        return triage_nearest_corecn(args)
    if args.recover_index_only:
        return recover_index(args)
    if args.merge_index_only:
        return merge_indices(args)
    if args.audit_xyz:
        return audit_xyz(args)
    if args.prescreen_only:
        return prescreen_only(args)
    return run_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
