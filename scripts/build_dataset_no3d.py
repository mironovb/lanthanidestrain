#!/usr/bin/env python3
"""Build the clean, deduplicated, no-3D ML dataset + a unique geometry-spec table.

This is stage 1 of the lanthanide dataset builder. It reads the raw SAFE
extraction exports, cleans them, deduplicates by *scientific content* (not
provenance), computes the distribution ratio target, attaches RDKit ligand
descriptors + ECFP fingerprints, metal descriptors and parsed experimental
condition features, drops provenance / target-leaking columns from the feature
table, and finally attaches a deterministic Architector geometry *plan* to every
row. No 3D coordinates are generated here -- that is stage 2
(``build_unique_geometries.py``), which consumes the ``geometry_specs.csv`` this
script writes.

The geometry plan is one unique physical complex per (metal, ligand,
inner-sphere anion): coordination number from the lanthanide contraction, donor
set from the ligand graph, stoichiometry by mass balance, nitrate-vs-water fill
from the acid. Temperature, contact time, phase ratio, D and provenance never
define a different 3D complex, so they are *not* part of the geometry key.

Usage:
    python scripts/build_dataset_no3d.py
    python scripts/build_dataset_no3d.py --limit 200          # smoke test on cleaned rows
    python scripts/build_dataset_no3d.py --force-cn 9         # legacy fixed CN
    python scripts/build_dataset_no3d.py --raw-dir data/raw   # alt input dir
    python scripts/build_dataset_no3d.py --with-3d-features   # also write the 3D-joined dataset
    python scripts/build_dataset_no3d.py --with-3d-features --compute-geometry-features

Outputs (under data/processed/):
    final_ml_dataset_no3d.parquet / _preview.csv / _info.json
    geometry_specs.csv               (one row per unique complex, for stage 2)
    geometry_dedup_mapping.csv       (dataset row -> geometry build_id)
    audit/invalid_smiles.csv
    audit/dedup_mapping.csv          (every raw row -> representative + reason)
    audit/dropped_columns.json

Optional stage 3 (only with --with-3d-features; the no-3D baseline above is left
byte-for-byte unchanged):
    final_ml_dataset_3d.parquet / _preview.csv   (baseline + clean 3D feature blocks)
    final_ml_dataset_3d_manifest.json            (input tables, join keys, feature
                                                  blocks, status + weight columns)
    sample_weights_3d.csv                        (training weights, keyed by safe_exp_id)
    feature_blocks/                              (physical/polyhedron tables,
                                                  image-native PI tensors, VR inputs,
                                                  ligand-PI control, xTB reference queue)

The 3D join is by build_id/geometry_key (geometry level) with safe_exp_id as the
stable row-level key. Only qc_class == OK geometries count as clean; every other
row keeps geometry_ok=False and receives no 3D feature values. Sample weights and
safe_exp_id are metadata and are excluded from is_feature_col().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

# Repo root is one level up: scripts/ -> repo root. Anchor every path here so the
# build runs correctly regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.chemistry.coordination import (  # noqa: E402
    LANTHANIDE_DESCRIPTORS,
    LANTHANIDE_SET,
    cn_for_Z,
    complex_build_id,
    inner_sphere_fill,
    plan_complex,
)
from src import geometry_schema as gschema  # noqa: E402
from src.geometry_schema import CLEAN_QC_CLASSES  # noqa: E402

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    import os
    import re
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "RDKit is not installed. Install it first:\n  python -m pip install rdkit\n"
    ) from exc

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR_CANDIDATES = [_REPO_ROOT / "raw_data", _REPO_ROOT / "data" / "raw"]
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
AUDIT_DIR = PROCESSED_DIR / "audit"

FINAL_DATASET_FILE = PROCESSED_DIR / "final_ml_dataset_no3d.parquet"
FINAL_PREVIEW_FILE = PROCESSED_DIR / "final_ml_dataset_no3d_preview.csv"
INFO_FILE = PROCESSED_DIR / "final_ml_dataset_no3d_info.json"
GEOMETRY_SPECS_FILE = PROCESSED_DIR / "geometry_specs.csv"
GEOMETRY_DEDUP_MAPPING_FILE = PROCESSED_DIR / "geometry_dedup_mapping.csv"
GEOMETRY_CONDITION_MAPPING_FILE = PROCESSED_DIR / "geometry_condition_mapping.csv"
INVALID_SMILES_FILE = AUDIT_DIR / "invalid_smiles.csv"
DEDUP_MAPPING_FILE = AUDIT_DIR / "dedup_mapping.csv"
DROPPED_COLUMNS_FILE = AUDIT_DIR / "dropped_columns.json"
INVALID_SPECS_FILE = AUDIT_DIR / "invalid_geometry_specs.csv"


# ---------------------------------------------------------------------------
# Column names / constants
# ---------------------------------------------------------------------------
TARGET = "log_D"
SMILES_COL = "LIGAND_SMILES"
CANONICAL_SMILES_COL = "canonical_smiles"
GROUP_COL = "extractant_group"
SPLIT_COL = "split"
METAL_COL = "metal"
RAW_ROW_COL = "__raw_row__"

COLUMN_ALIASES = {
    "metal": ["Metal_Name", "metal", "Metal", "metal_name", "Metal name"],
    "d_value": ["obsDvaluesValue", "D", "d", "D_value", "Distribution_Ratio", "distribution_ratio"],
    "extractant_smiles": ["Extractant_SMILES", "extractant_smiles", "SMILES", "smiles"],
    "extractant_name": ["Extractant_Name", "extractant_name", "Extractant", "extractant", "Ligand", "ligand"],
    "exp_id": ["exp_id", "Exp_ID", "id", "ID"],
    "acid_name": ["Acid_Name", "acid_name", "Acid", "acid", "aqueous_acid"],
}

NUMERIC_CONDITION_ALIASES = {
    "extractant_concentration_M": [
        "Extractant_Concentration_M", "Extractant Concentration M", "Extractant_Concentration",
        "extractant_concentration", "Ligand_Concentration_M", "ligand c.c./mM", "ligand concentration",
    ],
    "acid_concentration_M": [
        "Acid_Concentration_M", "Acid Concentration M", "Acid_Concentration", "acid_concentration",
        "acid c.c./M", "HNO3_Concentration", "HCl_Concentration", "Acidity",
    ],
    "metal_concentration_mM": [
        "Metal_Concentration_mM", "Metal Concentration mM", "Metal_Concentration", "metal_concentration",
        "Ln c.c./mM", "REE_Concentration_mM", "REE_Concentration",
    ],
    "temperature_C": ["obsTempsValue", "Temperature_C", "Temperature", "temperature", "Temp", "temp", "T"],
    "pH": ["pH", "PH", "ph"],
    "phase_ratio": ["Phase_Ratio", "phase_ratio", "O/A", "A/O", "organic/aqueous", "aqueous/organic"],
    "contact_time_min": [
        "Contact_Time_min", "Contact Time min", "Contact_Time", "contact_time",
        "Extraction_Time_min", "Shaking_Time_min", "time", "Time",
    ],
}

CATEGORICAL_CONDITION_ALIASES = {
    "acid": ["Acid", "acid", "Acid_Name", "acid_name", "aqueous_acid"],
    "diluent": ["Solvent_Name", "solvent_name", "Diluent", "diluent", "Diluent_Name",
                "Solvent", "solvent", "Organic_Solvent", "f_Solvent_Name"],
    "additive": ["Additive", "additive", "Modifier", "Synergist", "synergist",
                 "Phase_Modifier_Name", "Holdback_Agent_Name"],
}

CONDITION_FEATURE_MIN_NONMISSING_FRACTION = 0.02
CATEGORICAL_MAX_LEVELS = 40
MISSING_KEY_VALUE = "<MISSING>"

GEOMETRY_CONDITION_COLS = [
    "geom_cond__acid_class",
    "geom_cond__acid_strength_bin",
    "geom_cond__nitrate_activity",
    "geom_cond__pH_bin",
    "geom_cond__diluent_family",
    "geom_cond__modifier_class",
    "geom_cond__temperature_bin",
    "geom_cond__contact_time_bin",
    "geom_cond__shaking_time_bin",
    "geom_cond__phase_ratio_bin",
]

DEDUP_NUMERIC_ALIASES = {
    "D": ["D", "obsDvaluesValue"],
    "extractant_concentration_M": NUMERIC_CONDITION_ALIASES["extractant_concentration_M"],
    "acid_concentration_M": NUMERIC_CONDITION_ALIASES["acid_concentration_M"],
    "metal_concentration_mM": NUMERIC_CONDITION_ALIASES["metal_concentration_mM"],
    "temperature_C": NUMERIC_CONDITION_ALIASES["temperature_C"],
    "pH": NUMERIC_CONDITION_ALIASES["pH"],
    "phase_ratio": NUMERIC_CONDITION_ALIASES["phase_ratio"],
    "contact_time_min": ["Contact_Time_min", "Contact Time min", "Contact_Time", "contact_time"],
    "shaking_time_min": ["Shaking_Time_min", "Extraction_Time_min", "time", "Time"],
    "phase_modifier_concentration_M": ["Phase_Modifier_Concentration_M", "Modifier_Concentration_M"],
    "holdback_agent_concentration_M": ["Holdback_Agent_Concentration_M", "Holdback_Concentration_M"],
    "acid_concentration_organic_M": ["Acid_Concentration_Organic_M"],
    "f_metal_concentration_mM": ["f_Metal_Concentration_mM"],
    "radiolytic_dosage_kGy": ["Radiolytic_Dosage_kGy"],
    "vol_value": ["volValue"],
    "third_value": ["thirdValue"],
}

DEDUP_TEXT_ALIASES = {
    "metal_oxidation_state": ["Metal_Oxidation_state", "metal_oxidation_state", "oxidation_state"],
    "acid": CATEGORICAL_CONDITION_ALIASES["acid"],
    "diluent": CATEGORICAL_CONDITION_ALIASES["diluent"],
    "phase_modifier": ["Phase_Modifier_Name", "Modifier", "modifier", "Phase_Modifier"],
    "holdback_agent": ["Holdback_Agent_Name", "Holdback_Agent"],
    "vol_type": ["volType"],
    "third_type": ["thirdType"],
}

# Named RDKit ligand descriptors (the brief's explicit list).
RDKIT_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "MolLogP": Descriptors.MolLogP,
}
LIGAND_DESCRIPTOR_COLS = list(RDKIT_DESCRIPTOR_FUNCS.keys())

ECFP_BITS = 2048
ECFP_RADIUS = 2
ECFP_PREFIX = "ecfp_"
MORGAN_GENERATOR = GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)

METAL_DESCRIPTOR_COLS = ["Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal"]

# Provenance / metadata columns: excluded from the content-dedup key AND dropped
# from the final feature table (a different value here is NOT a different
# experiment, it is the SAFE export re-ingesting the same measurement).
PROVENANCE_COLS = {
    "source_file", RAW_ROW_COL, "exp_id", "safe_exp_id", "addition_date",
    "entry_author", "DOI", "doi", "doi_link", "publication", "paper",
    "comments_obsType", "comments_description", "comments", "obsDvalues",
    "obsTotalEnergiesUnit", "Extractant_inchi", "extractant_name",
    "Extractant_Name", "ini_comp", "Metal_Name",
}

# Substrings that mark a column as target-leaking or provenance and must never be
# a model feature. The explicit meta/target columns below are exempt.
LEAKAGE_KEYWORDS = (
    "log_d", "obsd", "distribution", "target", "true", "pred", "error", "err",
    "exp_id", "experiment_id", "publication", "paper", "doi", "source",
)

# Columns kept in the dataset as identifiers / target / geometry plan (never used
# as input features by a model, but needed downstream).
META_AND_PLAN_COLS = [
    METAL_COL, CANONICAL_SMILES_COL, SMILES_COL, "extractant_name",
    "D", TARGET, GROUP_COL, SPLIT_COL,
    # geometry plan attached per row (mirrors geometry_specs.csv)
    "build_id", "geometry_key", "metal_symbol", "metal_ox",
    "SMILES_FOR_ARCHITECTOR", "COORDLIST", "DONOR_TYPES", "DENTATE",
    "coreCN", "n_ligs", "inner_sphere_anion", "fill_ligand", "n_fill",
    "geometry_status", "geometry_note",
    # condition context that may affect interpretation of one reusable geometry
    "geometry_environment_key", *GEOMETRY_CONDITION_COLS,
]


# ---------------------------------------------------------------------------
# Optional 3D-feature enrichment (stage-3 join). All of this is inert unless the
# builder is run with --with-3d-features; the no-3D baseline outputs above are
# always produced first and never touched by this path.
# ---------------------------------------------------------------------------
REPORTS_DIR = _REPO_ROOT / "reports"

FINAL_DATASET_3D_FILE = PROCESSED_DIR / "final_ml_dataset_3d.parquet"
FINAL_PREVIEW_3D_FILE = PROCESSED_DIR / "final_ml_dataset_3d_preview.csv"
FEATURE_MANIFEST_FILE = PROCESSED_DIR / "final_ml_dataset_3d_manifest.json"
SAMPLE_WEIGHTS_FILE = PROCESSED_DIR / "sample_weights_3d.csv"
FEATURE_BLOCK_DIR = PROCESSED_DIR / "feature_blocks"

# Per-build_id geometry tables read (in order) to learn each complex's QC verdict
# and 3D-derived scalars. A build_id is treated as a *clean* geometry only when
# its qc_class == OK and it carries a usable .xyz path; the first usable clean
# table wins. Missing files are skipped, so the newer regenerated-accepted table
# slots in for free once build_unique_geometries.py has produced it.
def default_geometry_qc_tables() -> list[Path]:
    """QC sources in priority order, including isolated family rescue runs."""
    family_accepted = sorted(
        (REPORTS_DIR / "family_regeneration_runs").glob(
            "*/regenerated_fail_long_bond_accepted.csv"
        )
    )
    return [
        PROCESSED_DIR / "regenerated_fail_long_bond_accepted.csv",
        *family_accepted,
        REPORTS_DIR / "missing_geometry_rescue" / "regenerated_fail_long_bond_accepted.csv",
        REPORTS_DIR / "regenerated_fail_long_bond_accepted.csv",
        REPORTS_DIR / "geometry_ok_for_features.csv",
        REPORTS_DIR / "geometry_regenerate_fail_long_bond.csv",
        REPORTS_DIR / "geometry_borderline_review.csv",
    ]

# qc_class values that count as a usable clean 3D geometry are defined once in
# src.geometry_schema (CLEAN_QC_CLASSES, imported above) so producer and consumer
# can never disagree.

# All model-facing 3D feature columns carry this prefix so is_feature_col() can
# recognise them and so they can be blanket-nulled for rows without a clean
# geometry (no fake 3D values ever reach the model).
FEATURE_3D_PREFIX = "feat3d__"

# Training/loss metadata. These are deliberately NOT features (is_feature_col
# rejects the prefix) and are recomputed per-fold after a frozen split.
SAMPLE_WEIGHT_PREFIX = "sample_weight_"

# Identity / status / provenance columns carried into the enriched table only.
# safe_exp_id is the stable row-level join key (kept out of the baseline so that
# output stays byte-for-byte identical); the rest describe geometry provenance.
GEOMETRY_META_COLS = [
    "safe_exp_id",
    "geometry_ok",
    "geometry_qc_class",
    "geometry_source",
    "geometry_feature_build_id",
    "xyz_path",
    "mol2_path",
    "geometry_xtb_energy_eV",
    "feature_block_manifest",
    "complex_pi_image_index",
    "vr_graph_index",
    "ligand_pi_control_image_index",
]

# Built-in feature block derived straight from the nearest-coreCN QC scalars that
# already live in the geometry tables. These are honest measurements off the
# accepted geometry (first-shell donor distances + the gap to the next donor),
# i.e. feature-block priority 1/2 (complex scalar + coordination-polyhedron
# geometry). Raw complex energy is carried as meta, never as a feature: binding
# and strain energies require reference calculations and must not be faked from a
# single complex energy.
QC_SCALAR_FEATURE_MAP = {
    gschema.CORECN_MAX_DIST: "coreCN_max_donor_dist",
    gschema.NEXT_DONOR_DIST: "next_donor_dist",
    gschema.GAP_AFTER_CORECN: "coreCN_donor_gap",
}
BUILTIN_BLOCK_NAME = "polyhedron_scalars"


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------
def normalize_col_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip())


def find_column(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    lookup = {normalize_col_name(c).lower(): c for c in df.columns}
    for alias in aliases:
        hit = lookup.get(normalize_col_name(alias).lower())
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"Could not find any of {aliases}\nAvailable: {list(df.columns)}")
    return None


def find_existing_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        hit = lower.get(str(alias).strip().lower())
        if hit is not None:
            return hit
    return None


def normalize_text_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-", "--", "n/a", "na"}:
        return None
    return re.sub(r"\s+", " ", text)


def first_number_from_text(value: Any) -> float | None:
    text = normalize_text_value(value)
    if text is None:
        return None
    text = text.replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_concentration(value: Any, default_unit: str) -> float | None:
    """Parse '0.5 M', '10 mM', '0.01 mol/L' into the requested unit (M or mM).

    mg/L and ppm are NOT converted to molar (that needs a molecular weight); they
    are returned as the bare number so they remain an auditable numeric only.
    """
    text = normalize_text_value(value)
    if text is None:
        return None
    number = first_number_from_text(text)
    if number is None:
        return None
    lower = text.lower().replace("μ", "u").replace("µ", "u")

    if "mg/l" in lower or "mg l" in lower or "ppm" in lower:
        return number  # not safely convertible to molar
    if "mmol" in lower or re.search(r"\bmm\b", lower):
        value_m = number / 1000.0
    elif "umol" in lower or re.search(r"\bum\b", lower):
        value_m = number / 1_000_000.0
    elif "mol/l" in lower or re.search(r"(?<![a-z])m(?![a-z])", lower):
        value_m = number
    else:
        return number  # unit-less number; assume already in requested unit
    return value_m * 1000.0 if default_unit == "mM" else value_m


def parse_temperature_c(value: Any) -> float | None:
    number = first_number_from_text(value)
    if number is None:
        return None
    lower = (normalize_text_value(value) or "").lower()
    if "k" in lower and "°" not in lower and number > 100:
        return number - 273.15
    return number


def parse_phase_ratio(value: Any) -> float | None:
    text = normalize_text_value(value)
    if text is None:
        return None
    text = text.replace(" ", "")
    ratio = re.search(r"(\d*\.?\d+)[:/](\d*\.?\d+)", text)
    if ratio:
        denom = float(ratio.group(2))
        return float(ratio.group(1)) / denom if denom else None
    return first_number_from_text(text)


def parse_condition_series(series: pd.Series, feature_name: str) -> pd.Series:
    if feature_name.endswith("_concentration_M"):
        return series.map(lambda v: parse_concentration(v, "M"))
    if feature_name.endswith("_concentration_mM"):
        return series.map(lambda v: parse_concentration(v, "mM"))
    if feature_name == "temperature_C":
        return series.map(parse_temperature_c)
    if feature_name == "phase_ratio":
        return series.map(parse_phase_ratio)
    return series.map(first_number_from_text)


def normalize_key_text(value: Any) -> str:
    text = normalize_text_value(value)
    if text is None:
        return MISSING_KEY_VALUE
    text = text.lower().replace("μ", "u").replace("µ", "u")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key_number(value: Any, parser=first_number_from_text) -> str:
    number = parser(value)
    if number is None or not np.isfinite(number):
        return MISSING_KEY_VALUE
    return f"{float(number):.12g}"


def dedup_numeric_parser(component: str):
    if component.endswith("_concentration_M"):
        return lambda v: parse_concentration(v, "M")
    if component.endswith("_concentration_mM"):
        return lambda v: parse_concentration(v, "mM")
    if component == "temperature_C":
        return parse_temperature_c
    if component == "phase_ratio":
        return parse_phase_ratio
    return first_number_from_text


def build_content_key_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Build normalized, provenance-blind keys for true scientific duplicates."""
    key = pd.DataFrame(index=df.index)
    sources: dict[str, str | None] = {}

    key[METAL_COL] = df[METAL_COL].map(normalize_key_text)
    sources[METAL_COL] = METAL_COL
    key[CANONICAL_SMILES_COL] = df[CANONICAL_SMILES_COL].map(normalize_key_text)
    sources[CANONICAL_SMILES_COL] = CANONICAL_SMILES_COL

    for component, aliases in DEDUP_NUMERIC_ALIASES.items():
        if component == "D" and "D" in df.columns:
            series = df["D"]
            sources[component] = "D"
            key[component] = series.map(lambda v: normalize_key_number(v, first_number_from_text))
            continue

        col = find_existing_column(df, aliases)
        sources[component] = col
        if col is None:
            key[component] = MISSING_KEY_VALUE
            continue
        parser = dedup_numeric_parser(component)
        key[component] = df[col].map(lambda v: normalize_key_number(v, parser))

    for component, aliases in DEDUP_TEXT_ALIASES.items():
        col = find_existing_column(df, aliases)
        sources[component] = col
        if col is None:
            key[component] = MISSING_KEY_VALUE
            continue
        key[component] = df[col].map(normalize_key_text)

    return key, sources


def hash_content_key(row: pd.Series) -> str:
    payload = row.to_dict()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def acid_class(value: Any) -> str:
    text = normalize_key_text(value)
    if text == MISSING_KEY_VALUE:
        return "unknown"
    if any(tok in text for tok in ("hno3", "nitrate", "nitric")):
        return "nitrate"
    if "hclo4" in text or "perchlor" in text:
        return "perchlorate"
    if re.search(r"\bhcl\b|hydrochlor", text):
        return "chloride"
    if "h2so4" in text or "sulf" in text:
        return "sulfate"
    if any(tok in text for tok in ("oxalic", "malonic", "lactic", "tartaric", "citric", "carboxyl")):
        return "carboxylate"
    return "other_acid"


def concentration_bin(value: Any, parser=lambda v: parse_concentration(v, "M")) -> str:
    number = parser(value)
    if number is None or not np.isfinite(number):
        return "missing"
    if number < 0.01:
        return "trace"
    if number < 0.05:
        return "very_low"
    if number < 0.5:
        return "low"
    if number < 2.0:
        return "medium"
    if number < 6.0:
        return "high"
    return "very_high"


def nitrate_activity_bin(acid_name: Any, acid_concentration: Any) -> str:
    if acid_class(acid_name) != "nitrate":
        return "none"
    return concentration_bin(acid_concentration, lambda v: parse_concentration(v, "M"))


def pH_bin(value: Any) -> str:
    number = first_number_from_text(value)
    if number is None or not np.isfinite(number):
        return "missing"
    if number < 1:
        return "superacidic"
    if number < 3:
        return "strongly_acidic"
    if number < 6:
        return "weakly_acidic"
    if number < 8:
        return "near_neutral"
    return "basic"


def temperature_bin(value: Any) -> str:
    number = parse_temperature_c(value)
    if number is None or not np.isfinite(number):
        return "missing"
    if number < 15:
        return "cold"
    if number <= 30:
        return "ambient"
    if number <= 60:
        return "warm"
    return "hot"


def time_bin(value: Any) -> str:
    number = first_number_from_text(value)
    if number is None or not np.isfinite(number):
        return "missing"
    if number < 5:
        return "very_short"
    if number < 30:
        return "short"
    if number <= 120:
        return "medium"
    return "long"


def phase_ratio_bin(value: Any) -> str:
    number = parse_phase_ratio(value)
    if number is None or not np.isfinite(number):
        return "missing"
    if number < 0.5:
        return "aqueous_excess"
    if number <= 2.0:
        return "balanced"
    return "organic_excess"


def diluent_family(value: Any) -> str:
    text = normalize_key_text(value)
    if text == MISSING_KEY_VALUE:
        return "unknown"
    if any(tok in text for tok in ("dodecane", "kerosene", "isopar", "aliphatic", "tph", "hydrogenated")):
        return "aliphatic_hydrocarbon"
    if any(tok in text for tok in ("toluene", "benzene", "xylene", "tert-butylbenzene", "nphe")):
        return "aromatic"
    if any(tok in text for tok in ("chloroform", "dichloromethane", "ch3cl", "chlorinated")):
        return "chlorinated"
    if any(tok in text for tok in ("octanol", "decanol", "alcohol", "exxal")):
        return "alcohol_modifier"
    if any(tok in text for tok in ("nitrobenzene", "nitro")):
        return "nitroaromatic"
    if any(tok in text for tok in ("c4mim", "ionic", "tf2n")):
        return "ionic_liquid"
    return "other"


def modifier_class(value: Any) -> str:
    text = normalize_key_text(value)
    if text == MISSING_KEY_VALUE:
        return "none"
    if any(tok in text for tok in ("tbp", "toa", "phosph", "hdehp")):
        return "organophosphorus"
    if any(tok in text for tok in ("octanol", "decanol", "alcohol", "isodec")):
        return "alcohol"
    if "nano3" in text or "nitrate" in text:
        return "salt_nitrate"
    return "other_modifier"


def build_geometry_condition_context(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    sources: dict[str, str | None] = {}
    ctx = pd.DataFrame(index=df.index)

    acid_col = find_existing_column(df, CATEGORICAL_CONDITION_ALIASES["acid"])
    acid_conc_col = find_existing_column(df, NUMERIC_CONDITION_ALIASES["acid_concentration_M"])
    pH_col = find_existing_column(df, NUMERIC_CONDITION_ALIASES["pH"])
    diluent_col = find_existing_column(df, CATEGORICAL_CONDITION_ALIASES["diluent"])
    modifier_col = find_existing_column(df, ["Phase_Modifier_Name", "Modifier", "Synergist", "additive"])
    temperature_col = find_existing_column(df, NUMERIC_CONDITION_ALIASES["temperature_C"])
    contact_col = find_existing_column(df, ["Contact_Time_min", "Contact Time min", "Contact_Time", "contact_time"])
    shaking_col = find_existing_column(df, ["Shaking_Time_min", "Extraction_Time_min"])
    phase_ratio_col = find_existing_column(df, NUMERIC_CONDITION_ALIASES["phase_ratio"])

    sources.update({
        "acid": acid_col,
        "acid_concentration_M": acid_conc_col,
        "pH": pH_col,
        "diluent": diluent_col,
        "modifier": modifier_col,
        "temperature_C": temperature_col,
        "contact_time_min": contact_col,
        "shaking_time_min": shaking_col,
        "phase_ratio": phase_ratio_col,
    })

    acid = df[acid_col] if acid_col else pd.Series([None] * len(df), index=df.index)
    acid_conc = df[acid_conc_col] if acid_conc_col else pd.Series([None] * len(df), index=df.index)

    ctx["geom_cond__acid_class"] = acid.map(acid_class)
    ctx["geom_cond__acid_strength_bin"] = acid_conc.map(concentration_bin)
    ctx["geom_cond__nitrate_activity"] = [
        nitrate_activity_bin(a, c) for a, c in zip(acid, acid_conc)
    ]
    ctx["geom_cond__pH_bin"] = df[pH_col].map(pH_bin) if pH_col else "missing"
    ctx["geom_cond__diluent_family"] = df[diluent_col].map(diluent_family) if diluent_col else "unknown"
    ctx["geom_cond__modifier_class"] = df[modifier_col].map(modifier_class) if modifier_col else "none"
    ctx["geom_cond__temperature_bin"] = df[temperature_col].map(temperature_bin) if temperature_col else "missing"
    ctx["geom_cond__contact_time_bin"] = df[contact_col].map(time_bin) if contact_col else "missing"
    ctx["geom_cond__shaking_time_bin"] = df[shaking_col].map(time_bin) if shaking_col else "missing"
    ctx["geom_cond__phase_ratio_bin"] = df[phase_ratio_col].map(phase_ratio_bin) if phase_ratio_col else "missing"

    ctx["geometry_environment_key"] = ctx[GEOMETRY_CONDITION_COLS].agg(
        lambda row: "|".join(f"{col.removeprefix('geom_cond__')}={row[col]}" for col in GEOMETRY_CONDITION_COLS),
        axis=1,
    )
    return ctx, sources


def safe_dummy_name(prefix: str, value: Any) -> str:
    text = (normalize_text_value(value) or "missing").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return f"cond__{prefix}__{text[:60]}"


def clean_smiles(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "-", "nan", "none", "null"} else text


def clean_metal(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"\b(La|Ce|Pr|Nd|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu)\b", text)
    return match.group(1) if match else text


class suppress_rdkit_stderr:
    """Silence RDKit's noisy C++ stderr while parsing dirty SMILES."""

    def __enter__(self):
        self._saved = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self._saved, 2)
        os.close(self._saved)
        os.close(self._devnull)
        return False


def canonicalize_smiles(smiles: Any) -> str | None:
    """Canonicalise a SMILES; accept comma-lists only if they reduce to one mol."""
    if pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    candidates = [text]
    if text.endswith(","):
        candidates.append(text.rstrip(",").strip())
    if "," in text:
        candidates.extend(p.strip().strip(",") for p in text.split(",") if p.strip().strip(","))

    valid = []
    for candidate in dict.fromkeys(candidates):
        with suppress_rdkit_stderr():
            mol = Chem.MolFromSmiles(candidate)
        if mol is not None:
            valid.append(Chem.MolToSmiles(mol, canonical=True))

    unique = sorted(set(valid))
    return unique[0] if len(unique) == 1 else None


# ---------------------------------------------------------------------------
# Stage A: read + clean + deduplicate
# ---------------------------------------------------------------------------
def find_raw_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"--raw-dir not found: {explicit}")
        return explicit
    for cand in RAW_DIR_CANDIDATES:
        if cand.exists() and any(cand.glob("*.csv")):
            return cand
    raise SystemExit(f"No raw CSVs found in any of: {[str(c) for c in RAW_DIR_CANDIDATES]}")


def read_raw(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
        df.columns = [normalize_col_name(c) for c in df.columns]
        df["source_file"] = path.name
        frames.append(df)
        print(f"  read {path.name}: {len(df)} rows")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw[RAW_ROW_COL] = np.arange(len(raw))
    return raw


def clean_and_dedup(raw: pd.DataFrame, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Clean, filter to lanthanides, compute the target, and content-dedup.

    Returns (clean_df, dedup_mapping_df, counts). The mapping has one row per raw
    input row recording why it was kept or dropped and its representative.
    """
    metal_col = find_column(raw, COLUMN_ALIASES["metal"])
    d_col = find_column(raw, COLUMN_ALIASES["d_value"])
    smiles_col = find_column(raw, COLUMN_ALIASES["extractant_smiles"])
    name_col = find_column(raw, COLUMN_ALIASES["extractant_name"], required=False)
    exp_id_col = find_column(raw, COLUMN_ALIASES["exp_id"], required=False)

    df = raw.copy()
    df[METAL_COL] = df[metal_col].map(clean_metal)
    df["D"] = pd.to_numeric(df[d_col], errors="coerce")
    df[SMILES_COL] = df[smiles_col].map(clean_smiles)
    df["extractant_name"] = df[name_col].astype(str).str.strip() if name_col else ""
    # SAFE exp_id values are unique only *within* one metal export.  Prefix with
    # the source stem so the row-level join key is globally unique and stable
    # across rebuilds (e.g. Eu_SAFE:123 and Nd_SAFE:123 are distinct rows).
    source_stem = df["source_file"].astype("string").map(lambda value: Path(str(value)).stem)
    if exp_id_col:
        raw_exp_id = df[exp_id_col].astype("string").str.strip()
        missing_exp_id = raw_exp_id.isna() | raw_exp_id.eq("")
        raw_exp_id = raw_exp_id.mask(
            missing_exp_id,
            "row-" + df[RAW_ROW_COL].astype("string"),
        )
    else:
        raw_exp_id = "row-" + df[RAW_ROW_COL].astype("string")
    df["safe_exp_id"] = (source_stem + ":" + raw_exp_id).astype("string")

    counts: dict[str, Any] = {"raw_rows": int(len(df)), "raw_columns": int(raw.shape[1] - 2)}

    # Per-raw-row mapping records (reason filled as we drop / keep).
    mapping = pd.DataFrame({
        "original_row_index": df[RAW_ROW_COL].astype(int),
        "source_file": df["source_file"],
        "dedup_key": "",
        "dedup_key_components": "",
        "representative_row_index": pd.NA,
        "kept": False,
        "reason": "",
    }).set_index("original_row_index")

    def mark(mask_index, reason):
        mapping.loc[mask_index, "reason"] = reason

    # 1. Lanthanide filter.
    keep = df[METAL_COL].isin(LANTHANIDE_SET)
    mark(df.loc[~keep, RAW_ROW_COL], "dropped_non_lanthanide")
    df = df[keep].copy()
    counts["after_lanthanide_filter"] = int(len(df))

    # 2. Finite positive D -> log_D.
    keep = df["D"].notna() & (df["D"] > 0)
    mark(df.loc[~keep, RAW_ROW_COL], "dropped_invalid_D")
    df = df[keep].copy()
    df[TARGET] = np.log10(df["D"])
    counts["after_valid_D"] = int(len(df))

    # 3. Non-missing SMILES.
    keep = df[SMILES_COL].notna()
    mark(df.loc[~keep, RAW_ROW_COL], "dropped_missing_smiles")
    df = df[keep].copy()
    counts["after_smiles_present"] = int(len(df))

    if limit is not None:
        if limit <= 0:
            raise SystemExit("--limit must be a positive integer")
        keep_limited = pd.Series(False, index=df.index)
        keep_limited.loc[df.index[:limit]] = True
        mark(df.loc[~keep_limited, RAW_ROW_COL], "dropped_limit_smoke_test")
        df = df[keep_limited].copy()
    counts["after_limit"] = int(len(df))

    # 4. Exact full-row duplicates (identical across all original raw columns).
    #    Each duplicate maps to the first occurrence's raw index.
    orig_cols = [c for c in raw.columns if c not in {RAW_ROW_COL, "source_file"}]
    exact_dup = df.duplicated(subset=orig_cols, keep="first")
    first_idx = df.groupby(orig_cols, dropna=False, sort=False)[RAW_ROW_COL].transform("first")
    dup_labels = df.loc[exact_dup, RAW_ROW_COL].values
    mapping.loc[dup_labels, "reason"] = "dropped_exact_duplicate"
    mapping.loc[dup_labels, "representative_row_index"] = first_idx[exact_dup].astype(int).values
    df = df[~exact_dup].copy()
    counts["exact_duplicates_removed"] = int(exact_dup.sum())

    # 5. Canonicalise SMILES; drop invalid / ambiguous.
    df[CANONICAL_SMILES_COL] = df[SMILES_COL].map(canonicalize_smiles)
    invalid = df[CANONICAL_SMILES_COL].isna()
    if invalid.any():
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        df.loc[invalid, [SMILES_COL]].drop_duplicates().to_csv(INVALID_SMILES_FILE, index=False)
    else:
        INVALID_SMILES_FILE.unlink(missing_ok=True)
    mark(df.loc[invalid, RAW_ROW_COL], "dropped_invalid_smiles")
    df = df[~invalid].copy()
    counts["invalid_smiles_removed"] = int(invalid.sum())
    counts["after_valid_canonical_smiles"] = int(len(df))

    # 6. Content (provenance-blind) deduplication. Two rows that agree on metal +
    #    canonical ligand + D + normalized experimental conditions are the same
    #    measurement re-ingested from overlapping source files. Keep the first.
    content_key_frame, key_sources = build_content_key_frame(df)
    content_key = list(content_key_frame.columns)
    key_hash = content_key_frame.apply(hash_content_key, axis=1)
    df["_dedup_key"] = key_hash
    rep_idx = df.groupby("_dedup_key", sort=False)[RAW_ROW_COL].transform("first")
    content_dup = df.duplicated(subset="_dedup_key", keep="first")

    components_json = json.dumps(content_key, separators=(",", ":"))
    labels = df[RAW_ROW_COL].values
    mapping.loc[labels, "dedup_key"] = df["_dedup_key"].values
    mapping.loc[labels, "dedup_key_components"] = components_json
    mapping.loc[labels, "representative_row_index"] = rep_idx.astype(int).values
    mapping.loc[labels, "kept"] = (~content_dup).values
    mapping.loc[labels, "reason"] = np.where(
        content_dup.values, "dropped_content_duplicate", "kept_representative")

    df = df[~content_dup].drop(columns=["_dedup_key"]).reset_index(drop=True)
    counts["content_duplicates_removed"] = int(content_dup.sum())
    counts["deduplicated_rows"] = int(len(df))
    counts["content_dedup_key_sources"] = key_sources
    counts["content_dedup_excluded_provenance"] = sorted(PROVENANCE_COLS & set(raw.columns))
    counts["content_dedup_key_columns"] = content_key
    counts["detected_columns"] = {
        "metal": metal_col, "d_value": d_col, "extractant_smiles": smiles_col,
        "extractant_name": name_col, "exp_id": exp_id_col,
    }

    df[GROUP_COL] = df[CANONICAL_SMILES_COL]
    df[SPLIT_COL] = "unassigned"

    mapping_out = mapping.reset_index()
    return df, mapping_out, counts


# ---------------------------------------------------------------------------
# Stage B: ligand / metal / condition features
# ---------------------------------------------------------------------------
def ecfp_bits(mol: Chem.Mol) -> np.ndarray:
    bitvect = MORGAN_GENERATOR.GetFingerprint(mol)
    arr = np.zeros((ECFP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def build_ligand_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    unique = sorted(df[CANONICAL_SMILES_COL].dropna().astype(str).unique())
    if not unique:
        return df.copy(), {
            "unique_canonical_ligands": 0,
            "ecfp_bits": ECFP_BITS,
            "ecfp_radius": ECFP_RADIUS,
            "ligand_descriptors": LIGAND_DESCRIPTOR_COLS,
        }
    rows = []
    for idx, smiles in enumerate(unique, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(unique):
            print(f"  RDKit/ECFP {idx}/{len(unique)}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        row: dict[str, Any] = {CANONICAL_SMILES_COL: smiles}
        for name, fn in RDKIT_DESCRIPTOR_FUNCS.items():
            try:
                val = fn(mol)
                row[name] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
            except Exception:
                row[name] = np.nan
        ecfp = ecfp_bits(mol)
        row.update({f"{ECFP_PREFIX}{i}": int(v) for i, v in enumerate(ecfp)})
        rows.append(row)

    feats = pd.DataFrame(rows)
    merged = df.merge(feats, on=CANONICAL_SMILES_COL, how="left", validate="many_to_one")
    return merged, {
        "unique_canonical_ligands": int(len(unique)),
        "ecfp_bits": ECFP_BITS,
        "ecfp_radius": ECFP_RADIUS,
        "ligand_descriptors": LIGAND_DESCRIPTOR_COLS,
    }


def extract_condition_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_series: dict[str, pd.Series] = {}
    dropped: dict[str, str] = {}
    used_numeric, used_categorical = {}, {}
    min_nonmissing = max(1, math.ceil(CONDITION_FEATURE_MIN_NONMISSING_FRACTION * len(df)))

    for out_col, aliases in NUMERIC_CONDITION_ALIASES.items():
        col = find_existing_column(df, aliases)
        if col is None:
            continue
        used_numeric[out_col] = col
        parsed = pd.to_numeric(parse_condition_series(df[col], out_col), errors="coerce")
        parsed = parsed.replace([np.inf, -np.inf], np.nan)
        name = f"cond__{out_col}"
        if int(parsed.notna().sum()) < min_nonmissing:
            dropped[name] = f"too_few_parsed_values:{int(parsed.notna().sum())}"
            continue
        if int(parsed.nunique(dropna=True)) <= 1:
            dropped[name] = "constant_or_empty"
            continue
        feature_series[name] = parsed.astype(float)

    categorical_frames = []
    for out_col, aliases in CATEGORICAL_CONDITION_ALIASES.items():
        col = find_existing_column(df, aliases)
        if col is None:
            continue
        used_categorical[out_col] = col
        cleaned = df[col].map(normalize_text_value)
        if int(cleaned.notna().sum()) < min_nonmissing:
            dropped[f"cond__{out_col}"] = f"too_few_nonmissing:{int(cleaned.notna().sum())}"
            continue
        if int(cleaned.nunique(dropna=True)) <= 1:
            dropped[f"cond__{out_col}"] = "constant_categorical"
            continue
        if int(cleaned.nunique(dropna=True)) > CATEGORICAL_MAX_LEVELS:
            top = cleaned.value_counts(dropna=True).head(CATEGORICAL_MAX_LEVELS).index
            cleaned = cleaned.where(cleaned.isin(top), other="OTHER")
        dummies = pd.get_dummies(cleaned, dummy_na=False)
        dummies.columns = [safe_dummy_name(out_col, c) for c in dummies.columns]
        dummies = dummies.loc[:, dummies.nunique(dropna=True) > 1]
        if not dummies.empty:
            categorical_frames.append(dummies.astype(np.int8))

    numeric = pd.DataFrame(feature_series, index=df.index)
    frames = [numeric, *categorical_frames]
    features = pd.concat(frames, axis=1) if any(not f.empty for f in frames) else pd.DataFrame(index=df.index)
    features = features.loc[:, ~features.columns.duplicated()].copy()

    return features, {
        "condition_features": list(features.columns),
        "condition_feature_count": int(features.shape[1]),
        "numeric_alias_sources": used_numeric,
        "categorical_alias_sources": used_categorical,
        "dropped_condition_features": dropped,
        "min_nonmissing_required": min_nonmissing,
    }


def add_metal_and_condition_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    metal_feats = pd.DataFrame.from_dict(LANTHANIDE_DESCRIPTORS, orient="index")
    metal_feats.index.name = METAL_COL
    metal_feats = metal_feats.reset_index()
    merged = df.merge(metal_feats, on=METAL_COL, how="left", validate="many_to_one")

    cond_features, cond_report = extract_condition_features(df)
    merged = pd.concat([merged.reset_index(drop=True), cond_features.reset_index(drop=True)], axis=1)
    return merged, {"metal_descriptors": METAL_DESCRIPTOR_COLS, "condition_report": cond_report}


# ---------------------------------------------------------------------------
# Stage C: geometry plan (specs + per-row attachment)
# ---------------------------------------------------------------------------
def _build_id(Z: int, smiles: str, coordlist_json: str, dentate: int, core_cn: int,
              n_ligs: int, anion: str, fill: str, n_fill_val: int) -> str:
    """Stable id for one physical complex. Drives filenames + dedup + resume.

    Only chemistry that changes the 3D complex enters the hash: metal, ligand,
    donor set, denticity, CN, ligand count, inner-sphere anion and fill. D,
    temperature, time, row index, source file and DOI never enter it.
    """
    return complex_build_id(
        metal_Z=Z,
        ligand_smiles=smiles,
        coord_list=json.loads(coordlist_json),
        denticity=dentate,
        core_cn=core_cn,
        n_ligs=n_ligs,
        inner_sphere_anion=anion,
        fill_ligand=fill,
        n_fill_value=n_fill_val,
    )


def build_geometry_specs(
    df: pd.DataFrame, force_cn: int | None, max_ligs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """One geometry spec per unique buildable first-sphere complex.

    Also returns a dataset-row -> build_id mapping so stage-2 geometries can be
    merged back, and attaches the plan columns onto the dataset rows in place.
    """
    acid_col = find_existing_column(df, COLUMN_ALIASES["acid_name"])
    acid_conc_col = find_existing_column(df, NUMERIC_CONDITION_ALIASES["acid_concentration_M"])
    condition_ctx, condition_sources = build_geometry_condition_context(df)
    work = df[[CANONICAL_SMILES_COL, "Atomic Number_metal", METAL_COL]].copy()
    acid_values = df[acid_col] if acid_col is not None else pd.Series([None] * len(df), index=df.index)
    acid_conc_values = (
        df[acid_conc_col].map(lambda v: parse_concentration(v, "M"))
        if acid_conc_col is not None else pd.Series([None] * len(df), index=df.index)
    )
    work["inner_sphere_anion"] = [
        inner_sphere_fill(acid, conc) for acid, conc in zip(acid_values, acid_conc_values)
    ]
    work["geometry_key"] = (
        work["Atomic Number_metal"].astype("Int64").astype(str) + "|"
        + work[CANONICAL_SMILES_COL].astype(str) + "|" + work["inner_sphere_anion"]
    )

    spec_rows = []
    plan_by_key: dict[str, dict[str, Any]] = {}
    group_cols = ["Atomic Number_metal", CANONICAL_SMILES_COL, "inner_sphere_anion"]
    for (Z, smiles, anion), grp in work.groupby(group_cols, sort=False):
        if pd.isna(Z):
            continue
        # Representative acid for this anion bucket (only affects the fill species).
        acid_repr = "HNO3" if anion == "nitrate" else None
        spec = plan_complex(metal_Z=int(Z), ligand_smiles=str(smiles),
                            acid_name=acid_repr, core_cn=force_cn, max_ligs=max_ligs)
        ctx_grp = condition_ctx.loc[grp.index]
        acid_conc_grp = pd.to_numeric(acid_conc_values.loc[grp.index], errors="coerce")
        temp_col = condition_sources.get("temperature_C")
        temp_grp = (
            pd.to_numeric(df.loc[grp.index, temp_col].map(parse_temperature_c), errors="coerce")
            if temp_col else pd.Series(dtype=float)
        )
        coordlist_json = json.dumps(spec.coord_list)
        donor_json = json.dumps(spec.donor_types)
        build_id = _build_id(spec.metal_Z, spec.ligand_smiles, coordlist_json,
                             spec.denticity, spec.core_cn, spec.n_ligs,
                             spec.fill_ligand, spec.fill_ligand, spec.n_fill)
        geometry_key = grp["geometry_key"].iloc[0]
        status = "planned" if spec.valid else "invalid_no_donors"
        row = {
            "build_id": build_id,
            "reference": "",
            "Atomic Number_metal": spec.metal_Z,
            "metal_symbol": spec.metal_symbol,
            "metal_ox": spec.metal_ox,
            "canonical_smiles": spec.ligand_smiles,
            "SMILES_FOR_ARCHITECTOR": spec.ligand_smiles,
            "COORDLIST": coordlist_json,
            "DONOR_TYPES": donor_json,
            "DENTATE": spec.denticity,
            "coreCN": spec.core_cn,
            "n_ligs": spec.n_ligs,
            "inner_sphere_anion": spec.fill_ligand,
            "fill_ligand": spec.fill_ligand,
            "n_fill": spec.n_fill,
            "geometry_key": geometry_key,
            "rows_covered": int(len(grp)),
            "condition_variant_count": int(ctx_grp["geometry_environment_key"].nunique()),
            "condition_environment_examples": ";".join(
                sorted(ctx_grp["geometry_environment_key"].dropna().unique())[:5]
            ),
            "acid_concentration_M_min": None if acid_conc_grp.dropna().empty else float(acid_conc_grp.min()),
            "acid_concentration_M_max": None if acid_conc_grp.dropna().empty else float(acid_conc_grp.max()),
            "temperature_C_min": None if temp_grp.dropna().empty else float(temp_grp.min()),
            "temperature_C_max": None if temp_grp.dropna().empty else float(temp_grp.max()),
            "geometry_status": status,
            "geometry_note": spec.note,
            "xyz_path": "",
            "mol2_path": "",
            "sdf_path": "",
        }
        spec_rows.append(row)
        plan_by_key[geometry_key] = {
            "build_id": build_id, "metal_symbol": spec.metal_symbol,
            "metal_ox": spec.metal_ox, "SMILES_FOR_ARCHITECTOR": spec.ligand_smiles,
            "COORDLIST": coordlist_json, "DONOR_TYPES": donor_json,
            "DENTATE": spec.denticity, "coreCN": spec.core_cn, "n_ligs": spec.n_ligs,
            "inner_sphere_anion": spec.fill_ligand, "fill_ligand": spec.fill_ligand,
            "n_fill": spec.n_fill, "geometry_status": status, "geometry_note": spec.note,
        }

    specs = pd.DataFrame(spec_rows)

    # Attach the plan to each dataset row via the geometry_key.
    df = df.copy()
    for col in ["geometry_environment_key", *GEOMETRY_CONDITION_COLS]:
        df[col] = condition_ctx[col].values
    df["inner_sphere_anion_key"] = work["inner_sphere_anion"].values
    df["geometry_key"] = work["geometry_key"].values
    plan_df = pd.DataFrame.from_dict(plan_by_key, orient="index")
    plan_df.index.name = "geometry_key"
    plan_df = plan_df.reset_index()
    df = df.merge(plan_df, on="geometry_key", how="left", validate="many_to_one")
    df = df.drop(columns=["inner_sphere_anion_key"])

    mapping = pd.DataFrame({
        "dataset_row_index": np.arange(len(df)),
        "safe_exp_id": df["safe_exp_id"],
        "metal": df[METAL_COL],
        "canonical_smiles": df[CANONICAL_SMILES_COL],
        "inner_sphere_anion": df["inner_sphere_anion"],
        "geometry_key": df["geometry_key"],
        "build_id": df["build_id"],
        "geometry_environment_key": df["geometry_environment_key"],
    })
    for col in GEOMETRY_CONDITION_COLS:
        mapping[col] = df[col]

    valid = specs[specs["geometry_status"] == "planned"]
    report = {
        "unique_geometry_specs": int(len(specs)),
        "planned_specs": int(len(valid)),
        "invalid_no_donor_specs": int((specs["geometry_status"] == "invalid_no_donors").sum()),
        "coreCN_distribution": {int(k): int(v) for k, v in valid["coreCN"].value_counts().sort_index().items()},
        "inner_sphere_anion_distribution": valid["inner_sphere_anion"].value_counts().to_dict(),
        "acid_column_used": acid_col,
        "acid_concentration_column_used": acid_conc_col,
        "condition_context_sources": condition_sources,
        "condition_environment_count": int(condition_ctx["geometry_environment_key"].nunique()),
        "condition_context_columns": ["geometry_environment_key", *GEOMETRY_CONDITION_COLS],
        "nitrate_activity_distribution": condition_ctx["geom_cond__nitrate_activity"].value_counts().to_dict(),
        "force_cn": force_cn,
        "pairs_differing_from_CN9": int((valid["Atomic Number_metal"].map(cn_for_Z) != 9).sum()),
    }
    return df, specs, mapping, report


def freeze_plan_to_existing_geometry_index(
    df: pd.DataFrame,
    specs: pd.DataFrame,
    mapping: pd.DataFrame,
    geom_report: dict[str, Any],
    index_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reuse the frozen stage-2 template for complexes already in the index.

    ``plan_complex()`` evolves as chemistry policies improve, so a later stage-1
    rebuild can legitimately produce a different build_id/COORDLIST/CN for the
    same stable geometry_key.  When the user explicitly asks to assemble a
    dataset from *existing* accepted structures, the historical stage-2 index is
    the source of truth for those frozen template fields.  New geometry keys,
    if any, retain the current plan.
    """
    if not index_path.exists():
        geom_report["existing_geometry_plan_freeze"] = {
            "status": "index_missing", "index": str(index_path), "matched_specs": 0,
        }
        return df, specs, mapping, geom_report

    index = pd.read_csv(index_path, low_memory=False, dtype={"build_id": "string"})
    required = {
        "build_id", "Atomic Number_metal", "SMILES_FOR_ARCHITECTOR",
        "inner_sphere_anion",
    }
    if not required.issubset(index.columns):
        missing = sorted(required - set(index.columns))
        raise ValueError(f"{index_path} missing frozen-plan columns: {missing}")

    index["geometry_key"] = (
        pd.to_numeric(index["Atomic Number_metal"], errors="coerce").astype("Int64").astype("string")
        + "|" + index["SMILES_FOR_ARCHITECTOR"].astype("string")
        + "|" + index["inner_sphere_anion"].astype("string")
    )
    if index["geometry_key"].isna().any() or index["geometry_key"].duplicated().any():
        raise ValueError(f"{index_path} does not define one frozen row per geometry_key")

    frozen_fields = [
        col for col in [
            "build_id", "Atomic Number_metal", "metal_symbol", "metal_ox",
            "SMILES_FOR_ARCHITECTOR", "COORDLIST", "DONOR_TYPES", "DENTATE",
            "coreCN", "n_ligs", "inner_sphere_anion", "fill_ligand", "n_fill",
        ] if col in index.columns
    ]
    frozen = index.set_index("geometry_key")[frozen_fields]
    matched_spec_mask = specs["geometry_key"].astype("string").isin(frozen.index)

    def apply_frozen(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        keys = out["geometry_key"].astype("string")
        matched = keys.isin(frozen.index)
        for col in frozen_fields:
            if col not in out.columns:
                continue
            values = keys.map(frozen[col])
            out.loc[matched, col] = values[matched].to_numpy()
        return out

    df = apply_frozen(df)
    specs = apply_frozen(specs)
    if "build_id" in mapping.columns:
        map_keys = mapping["geometry_key"].astype("string")
        matched = map_keys.isin(frozen.index)
        mapped_ids = map_keys.map(frozen["build_id"])
        mapping = mapping.copy()
        mapping.loc[matched, "build_id"] = mapped_ids[matched].to_numpy()

    valid = specs[specs["geometry_status"] == "planned"]
    geom_report = dict(geom_report)
    geom_report["coreCN_distribution"] = {
        int(k): int(v) for k, v in valid["coreCN"].value_counts().sort_index().items()
    }
    geom_report["pairs_differing_from_CN9"] = int((valid["coreCN"] != 9).sum())
    geom_report["existing_geometry_plan_freeze"] = {
        "status": "applied",
        "index": str(index_path),
        "index_rows": int(len(index)),
        "matched_specs": int(matched_spec_mask.sum()),
        "new_unmatched_specs": int((~matched_spec_mask).sum()),
        "frozen_fields": frozen_fields,
    }
    return df, specs, mapping, geom_report


# ---------------------------------------------------------------------------
# Stage D: drop leakage/provenance + select final columns
# ---------------------------------------------------------------------------
def is_feature_col(col: str) -> bool:
    # Training/loss weights are metadata, never model inputs -- reject first so a
    # future weight column can never accidentally match a feature rule.
    if col.startswith(SAMPLE_WEIGHT_PREFIX):
        return False
    return (
        col.startswith(ECFP_PREFIX)
        or col.startswith("cond__")
        or col.startswith(FEATURE_3D_PREFIX)
        or col in LIGAND_DESCRIPTOR_COLS
        or col in METAL_DESCRIPTOR_COLS
    )


def feature_sort_key(col: str) -> tuple[int, int | str]:
    if col in LIGAND_DESCRIPTOR_COLS:
        return (0, LIGAND_DESCRIPTOR_COLS.index(col))
    if col in METAL_DESCRIPTOR_COLS:
        return (1, METAL_DESCRIPTOR_COLS.index(col))
    if col.startswith("cond__"):
        return (2, col)
    if col.startswith(ECFP_PREFIX):
        suffix = col.removeprefix(ECFP_PREFIX)
        return (3, int(suffix) if suffix.isdigit() else suffix)
    if col.startswith(FEATURE_3D_PREFIX):
        return (4, col)
    return (5, col)


def select_final_columns(
    df: pd.DataFrame, extra_keep: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta_present = [c for c in META_AND_PLAN_COLS if c in df.columns]
    extra_present = [c for c in (extra_keep or []) if c in df.columns]
    feature_cols = sorted((c for c in df.columns if is_feature_col(c)), key=feature_sort_key)
    keep = list(dict.fromkeys(meta_present + extra_present + feature_cols))

    dropped: dict[str, str] = {}
    for col in df.columns:
        if col in keep:
            continue
        low = str(col).lower()
        if col in PROVENANCE_COLS:
            dropped[col] = "provenance"
        elif any(k in low for k in LEAKAGE_KEYWORDS):
            dropped[col] = "leakage_keyword"
        else:
            dropped[col] = "non_feature_raw_column"

    final = df[keep].copy()
    report = {
        "feature_columns": int(len(feature_cols)),
        "meta_and_plan_columns": meta_present,
        "dropped_columns": dropped,
        "dropped_column_count": len(dropped),
    }
    return final, report


# ---------------------------------------------------------------------------
# Stage E (optional): join clean 3D-derived feature blocks
# ---------------------------------------------------------------------------
def load_geometry_qc_index(
    paths: list[Path],
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Read + merge per-build_id geometry tables into one QC index.

    A build_id may appear in several tables (e.g. the OK queue and, later, the
    regenerated-accepted table). Each table is reconciled to the canonical schema
    (src.geometry_schema.normalize_qc_table) *before* the concat -- so the
    accepted table's ``accepted_*`` paths and missing ``xyz_exists`` never get
    silently blocked once mixed with the reports tables -- then we keep the row
    with the *best* verdict so a build rescued by regeneration is reported clean.

    Returns the de-duplicated index (empty if no table exists), the list of files
    used, and a per-file record of every schema fallback applied (empty list =
    the table already matched the canonical schema), so the reconciliation is
    auditable in the run manifest instead of being an invisible guess.
    """
    frames: list[pd.DataFrame] = []
    used: list[str] = []
    fallbacks: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            source_label = str(path.resolve().relative_to(_REPO_ROOT.resolve()))
        except ValueError:
            source_label = path.name
        # Preserve hash-like IDs exactly. Reading first and casting afterward is
        # too late for an all-numeric ID with leading zeroes.
        try:
            table = pd.read_csv(
                path, low_memory=False, dtype={gschema.BUILD_ID: "string"}
            )
        except EmptyDataError:
            fallbacks[source_label] = ["skipped empty CSV"]
            continue
        if gschema.BUILD_ID not in table.columns:
            fallbacks[source_label] = [f"skipped: missing {gschema.BUILD_ID}"]
            continue
        table, applied = gschema.normalize_qc_table(table)
        # geometry_key is stable across planning-policy revisions, whereas
        # build_id intentionally changes when COORDLIST/CN/stoichiometry changes.
        # Older QC/accepted tables often omitted geometry_key; reconstruct it
        # from their frozen complex identity so existing accepted geometries can
        # still be attached without pretending they have the newly planned ID.
        if "geometry_key" not in table.columns:
            table["geometry_key"] = pd.array([pd.NA] * len(table), dtype="string")
        else:
            table["geometry_key"] = table["geometry_key"].astype("string")
        smiles = pd.Series(pd.NA, index=table.index, dtype="string")
        for candidate in ("SMILES_FOR_ARCHITECTOR", "smiles_for_architector_used"):
            if candidate in table.columns:
                smiles = smiles.fillna(table[candidate].astype("string"))
        if {
            "Atomic Number_metal", "inner_sphere_anion",
        }.issubset(table.columns):
            metal_z = pd.to_numeric(table["Atomic Number_metal"], errors="coerce").astype("Int64").astype("string")
            anion = table["inner_sphere_anion"].astype("string")
            derived_key = metal_z + "|" + smiles + "|" + anion
            missing_key = table["geometry_key"].isna() | table["geometry_key"].str.strip().eq("")
            n_derived = int((missing_key & derived_key.notna()).sum())
            if n_derived:
                table.loc[missing_key, "geometry_key"] = derived_key[missing_key]
                applied.append(f"derived geometry_key for {n_derived} rows")
        # Keep a canonical extension dtype across input formats/tables.
        table[gschema.BUILD_ID] = table[gschema.BUILD_ID].astype("string")
        table["geometry_source"] = source_label
        frames.append(table)
        used.append(str(path))
        fallbacks[source_label] = applied
    if not frames:
        return pd.DataFrame(), used, fallbacks
    idx = pd.concat(frames, ignore_index=True, sort=False)
    if gschema.QC_CLASS not in idx.columns:
        idx[gschema.QC_CLASS] = pd.NA
    if gschema.XYZ_PATH not in idx.columns:
        idx[gschema.XYZ_PATH] = pd.NA
    xyz_exists = idx[gschema.XYZ_EXISTS].astype("string").str.lower().isin(["true", "1", "1.0"])
    xyz_path = idx[gschema.XYZ_PATH].astype("string").str.strip()
    usable_xyz = xyz_exists & xyz_path.notna() & (xyz_path != "")
    # Prefer clean verdicts with a usable path, then clean verdicts, then earlier
    # tables (stable sort keeps source-table priority).
    idx["_clean_rank"] = idx[gschema.QC_CLASS].astype("string").isin(CLEAN_QC_CLASSES).astype(int)
    idx["_usable_clean_rank"] = (idx["_clean_rank"].astype(bool) & usable_xyz).astype(int)
    idx = (
        idx.sort_values(["_usable_clean_rank", "_clean_rank"], ascending=False, kind="stable")
        .drop_duplicates(gschema.BUILD_ID, keep="first")
        .drop(columns=["_usable_clean_rank", "_clean_rank"])
        .reset_index(drop=True)
    )
    return idx, used, fallbacks


def parse_feature_block_specs(specs: list[str] | None) -> list[dict[str, Any]]:
    """Parse ``--feature-block name:path[:join_key[:gating]]`` CLI arguments.

    join_key defaults to build_id (a geometry-level block). gating defaults to
    "gated" (values are nulled for rows without a clean complex geometry); pass
    "ungated" for blocks that do not depend on the complex passing QC, e.g. the
    ligand-only persistence-image control block.
    """
    parsed: list[dict[str, Any]] = []
    for spec in specs or []:
        parts = spec.split(":")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise SystemExit(
                f"--feature-block must be name:path[:join_key[:gating]], got {spec!r}")
        name, path = parts[0], parts[1]
        join_key = parts[2] if len(parts) > 2 and parts[2] else "build_id"
        gating = parts[3] if len(parts) > 3 and parts[3] else "gated"
        if join_key not in {"build_id", "geometry_key", "safe_exp_id"}:
            raise SystemExit(
                f"--feature-block join_key must be build_id/geometry_key/safe_exp_id, got {join_key!r}")
        if gating not in {"gated", "ungated"}:
            raise SystemExit(f"--feature-block gating must be gated/ungated, got {gating!r}")
        parsed.append({
            "name": name, "path": Path(path), "join_key": join_key,
            "gated": gating == "gated",
        })
    return parsed


def attach_geometry_status(
    enriched: pd.DataFrame, qc_index: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach geometry_ok / qc_class / source / paths and the built-in scalar block.

    Prefer the QC row selected by stable geometry_key, falling back to build_id
    for legacy/fixture tables. This preserves the accepted structure's frozen
    build_id when current planning policy has changed. Only qc_class == OK with
    an existing .xyz counts as clean; every other row keeps geometry_ok=False.
    """
    n = len(enriched)
    enriched["geometry_ok"] = False
    enriched["geometry_qc_class"] = pd.array([pd.NA] * n, dtype="string")
    enriched["geometry_source"] = pd.array([pd.NA] * n, dtype="string")
    enriched["geometry_feature_build_id"] = pd.array([pd.NA] * n, dtype="string")
    enriched["xyz_path"] = pd.array([pd.NA] * n, dtype="string")
    enriched["mol2_path"] = pd.array([pd.NA] * n, dtype="string")
    enriched["geometry_xtb_energy_eV"] = pd.array([pd.NA] * n, dtype="Float64")
    scalar_cols: list[str] = []

    report: dict[str, Any] = {
        "clean_geometry_rows": 0, "rows_without_clean_geometry": n,
        "qc_class_distribution": {}, "builtin_scalar_features": [],
    }
    if qc_index.empty or "build_id" not in enriched.columns:
        report["note"] = "no geometry QC index found; all rows geometry_ok=False"
        return enriched, report

    idx = qc_index.set_index(gschema.BUILD_ID)
    bids = enriched["build_id"].astype("string")
    chosen_ids = bids.copy()

    # Prefer the best QC row for the stable scientific geometry_key. This is
    # essential after a planning-policy revision changes build_id: the accepted
    # geometry remains real, but must retain its own frozen build_id provenance.
    if "geometry_key" in enriched.columns and "geometry_key" in qc_index.columns:
        candidates = qc_index.copy()
        candidate_clean = candidates[gschema.QC_CLASS].astype("string").isin(CLEAN_QC_CLASSES)
        candidate_exists = candidates[gschema.XYZ_EXISTS].astype("string").str.lower().isin(
            ["true", "1", "1.0"]
        )
        candidate_path = candidates[gschema.XYZ_PATH].astype("string").str.strip()
        candidates["_usable_clean_rank"] = (
            candidate_clean & candidate_exists & candidate_path.notna() & candidate_path.ne("")
        ).astype(int)
        candidates["_clean_rank"] = candidate_clean.astype(int)
        by_key = (
            candidates.dropna(subset=["geometry_key"])
            .sort_values(["_usable_clean_rank", "_clean_rank"], ascending=False, kind="stable")
            .drop_duplicates("geometry_key", keep="first")
            .set_index("geometry_key")[gschema.BUILD_ID]
        )
        key_choice = enriched["geometry_key"].astype("string").map(by_key).astype("string")
        chosen_ids = key_choice.fillna(chosen_ids)

    present = chosen_ids.isin(idx.index)

    def lookup(col: str) -> pd.Series:
        if col not in idx.columns:
            return pd.Series(pd.NA, index=enriched.index, dtype="object")
        mapped = chosen_ids.map(idx[col])
        mapped.index = enriched.index
        return mapped

    qc_class = lookup(gschema.QC_CLASS).astype("string")
    xyz_exists_raw = lookup(gschema.XYZ_EXISTS)
    xyz_exists = xyz_exists_raw.astype("string").str.lower().isin(["true", "1", "1.0"])
    # If the table has no xyz_exists column, do not block on it. (After
    # normalize_qc_table every loaded table carries one; this guards direct
    # callers that pass a hand-built index.)
    if gschema.XYZ_EXISTS not in idx.columns:
        xyz_exists = present
    matched_ok = present & qc_class.isin(CLEAN_QC_CLASSES)
    clean = matched_ok & xyz_exists.fillna(False)

    enriched["geometry_qc_class"] = qc_class.where(present, other=pd.NA)
    enriched.loc[present, "geometry_source"] = lookup("geometry_source")[present].astype("string")
    enriched["geometry_feature_build_id"] = chosen_ids.where(present, other=pd.NA)
    enriched["xyz_path"] = lookup(gschema.XYZ_PATH).astype("string")
    enriched["mol2_path"] = lookup(gschema.MOL2_PATH).astype("string")
    enriched["geometry_xtb_energy_eV"] = pd.to_numeric(lookup(gschema.ENERGY_EV), errors="coerce")
    enriched["geometry_ok"] = clean.fillna(False).to_numpy()

    # Built-in polyhedron scalar block from the QC distances (clean rows only).
    for src, feat in QC_SCALAR_FEATURE_MAP.items():
        col = f"{FEATURE_3D_PREFIX}{BUILTIN_BLOCK_NAME}__{feat}"
        values = pd.to_numeric(lookup(src), errors="coerce")
        enriched[col] = values.where(enriched["geometry_ok"], other=np.nan)
        scalar_cols.append(col)

    report["clean_geometry_rows"] = int(enriched["geometry_ok"].sum())
    report["rows_without_clean_geometry"] = int((~enriched["geometry_ok"]).sum())
    report["qc_class_distribution"] = {
        str(k): int(v) for k, v in qc_class[present].value_counts().items()
    }
    report["rows_matched_by_planned_build_id"] = int((present & chosen_ids.eq(bids)).sum())
    report["rows_matched_by_geometry_key"] = int((present & chosen_ids.ne(bids)).sum())
    report["builtin_scalar_features"] = scalar_cols

    # Soft invariant: if OK verdicts matched dataset rows but the xyz gate wiped
    # out *every* one of them, that is the fingerprint of schema drift (a missing
    # or mis-parsed xyz_exists), not real chemistry. Surface it loudly instead of
    # silently shipping an all-False geometry_ok column.
    n_matched_ok = int(matched_ok.fillna(False).sum())
    if n_matched_ok and report["clean_geometry_rows"] == 0:
        report["schema_drift_warning"] = (
            f"{n_matched_ok} rows matched an OK geometry but 0 passed the "
            f"xyz-exists gate; check xyz_exists / xyz_path in the QC tables."
        )
        print(f"  WARNING: {report['schema_drift_warning']}")
    return enriched, report


def attach_feature_blocks(
    enriched: pd.DataFrame, blocks: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Left-join each external feature-block table by its declared key.

    Missing files / missing keys never fabricate values: absent blocks are
    recorded with present=False and contribute no columns. Gated blocks have
    their values nulled for rows without a clean complex geometry.
    """
    manifest: list[dict[str, Any]] = []
    for block in blocks:
        name, path, key, gated = block["name"], block["path"], block["join_key"], block["gated"]
        entry: dict[str, Any] = {
            "name": name, "source": str(path), "join_key": key, "gated": gated,
            "present": False, "matched_rows": 0, "columns": [],
        }
        if not path.exists():
            entry["status"] = "missing_file"
            manifest.append(entry)
            continue
        if key not in enriched.columns:
            entry["status"] = f"join_key_absent:{key}"
            manifest.append(entry)
            continue
        table = (pd.read_parquet(path) if path.suffix == ".parquet"
                 else pd.read_csv(path, low_memory=False))
        if key not in table.columns:
            entry["status"] = f"join_key_absent_in_block:{key}"
            manifest.append(entry)
            continue
        value_cols = [c for c in table.columns if c != key]
        renamed = {c: f"{FEATURE_3D_PREFIX}{name}__{c}" for c in value_cols}
        table = table[[key, *value_cols]].rename(columns=renamed).drop_duplicates(subset=key)
        before = set(enriched.columns)
        enriched = enriched.merge(table, on=key, how="left", validate="many_to_one")
        new_cols = [c for c in enriched.columns if c not in before]
        if gated and new_cols and "geometry_ok" in enriched.columns:
            mask = ~enriched["geometry_ok"].fillna(False).to_numpy()
            enriched.loc[mask, new_cols] = np.nan
        matched_rows = (
            int(enriched[new_cols].notna().any(axis=1).sum()) if new_cols else 0
        )
        entry.update({
            "present": True, "status": "joined",
            "matched_rows": matched_rows,
            "columns": new_cols,
        })
        manifest.append(entry)
    return enriched, manifest


def build_feature_block_manifest_column(
    enriched: pd.DataFrame, block_names: list[str],
) -> pd.Series:
    """Per-row ';'-joined list of feature blocks that contributed a value."""
    per_row = []
    block_cols = {
        name: [c for c in enriched.columns if c.startswith(f"{FEATURE_3D_PREFIX}{name}__")]
        for name in block_names
    }
    for _, row in enriched[[c for cols in block_cols.values() for c in cols]].iterrows():
        present = [name for name, cols in block_cols.items()
                   if cols and row[cols].notna().any()]
        per_row.append(";".join(present))
    return pd.Series(per_row, index=enriched.index, dtype="string")


def compute_sample_weights(enriched: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inverse metal-frequency training weights, normalised to mean 1.

    Weights are metadata, not features. They are computed here over the full
    table for convenience; the correct workflow recomputes them per-fold after a
    frozen train/val/test split (documented in the manifest). Adjacent-pair
    inverse-frequency weighting for a log_SF selectivity target needs matched
    condition pairs that this single-metal-per-row table does not carry, so it is
    intentionally left for the selectivity builder rather than faked here.
    """
    report: dict[str, Any] = {}
    col = f"{SAMPLE_WEIGHT_PREFIX}inv_metal_freq"
    metals = enriched[METAL_COL].astype("string")
    freq = metals.map(metals.value_counts())
    raw = 1.0 / freq.astype(float)
    mean = raw.mean()
    weights = raw / mean if mean and not math.isnan(mean) else pd.Series(1.0, index=enriched.index)
    enriched[col] = weights.to_numpy()
    report = {
        "column": col,
        "scheme": "inverse_metal_frequency",
        "normalisation": "mean=1",
        "computed_over": "full_table (recompute per-fold after frozen split)",
        "metal_frequencies": {str(k): int(v) for k, v in metals.value_counts().items()},
        "not_implemented": {
            "sample_weight_inv_adjacent_pair_freq":
                "needs matched-condition metal pairs + log_SF target (selectivity builder)",
        },
    }
    return enriched, report


def build_and_write_3d_dataset(
    clean: pd.DataFrame, args, geom_report: dict[str, Any],
) -> dict[str, Any]:
    """Build + write the optional 3D-enriched dataset (baseline is untouched)."""
    qc_tables = ([Path(p) for p in args.geometry_index]
                 if args.geometry_index else default_geometry_qc_tables())
    qc_index, qc_files, qc_fallbacks = load_geometry_qc_index(qc_tables)
    blocks = parse_feature_block_specs(args.feature_block)
    generated_artifacts: dict[str, Any] | None = None

    if args.compute_geometry_features:
        from src.geometry_features import build_geometry_feature_artifacts

        print("Computing feature blocks from QC-accepted existing geometries ...")
        generated_artifacts = build_geometry_feature_artifacts(
            qc_index,
            clean,
            repo_root=_REPO_ROOT,
            output_dir=Path(args.feature_output_dir),
            pi_resolution=int(args.pi_resolution),
            vr_max_edge=float(args.vr_max_edge),
        )
        # Generated blocks take priority, while repeatable user-supplied blocks
        # can still add a later project-specific representation.
        blocks = [*generated_artifacts["block_specs"], *blocks]

    enriched = clean.copy()
    enriched, geom_status_report = attach_geometry_status(enriched, qc_index)
    enriched, block_manifest = attach_feature_blocks(enriched, blocks)
    if generated_artifacts is not None:
        row_assets = generated_artifacts["row_asset_columns"]
        enriched = enriched.merge(
            row_assets,
            on="safe_exp_id",
            how="left",
            validate="one_to_one",
        )

    block_names = [BUILTIN_BLOCK_NAME] + [b["name"] for b in blocks]
    enriched["feature_block_manifest"] = build_feature_block_manifest_column(enriched, block_names)
    enriched, weight_report = compute_sample_weights(enriched)

    weight_cols = [c for c in enriched.columns if c.startswith(SAMPLE_WEIGHT_PREFIX)]
    extra_keep = GEOMETRY_META_COLS + weight_cols
    final_3d, column_report = select_final_columns(enriched, extra_keep=extra_keep)

    feat3d_cols = [c for c in final_3d.columns if c.startswith(FEATURE_3D_PREFIX)]

    try:
        final_3d.to_parquet(FINAL_DATASET_3D_FILE, index=False)
        parquet_3d_ok = True
    except Exception as exc:  # pragma: no cover
        print(f"  WARNING: 3D parquet write failed ({exc}); CSV preview still written.")
        parquet_3d_ok = False
    final_3d.head(200).to_csv(FINAL_PREVIEW_3D_FILE, index=False)

    # Weights written separately too, keyed by the stable row-level join key.
    weight_out = final_3d[["safe_exp_id", "build_id", "geometry_key", *weight_cols]].copy()
    weight_out.insert(0, "dataset_row_index", np.arange(len(weight_out)))
    weight_out.to_csv(SAMPLE_WEIGHTS_FILE, index=False)

    manifest = {
        "description": "Optional 3D-enriched ML dataset. Baseline no-3D dataset is "
                       "unchanged; this file adds clean geometry-derived feature blocks.",
        "input_tables": {
            "baseline_dataset": str(FINAL_DATASET_FILE),
            "geometry_qc_tables_used": qc_files,
            "geometry_qc_schema_fallbacks": qc_fallbacks,
            "geometry_dedup_mapping": str(GEOMETRY_DEDUP_MAPPING_FILE),
            "external_feature_blocks": [b["source"] for b in block_manifest],
        },
        "join_keys": {
            "geometry_level": ["build_id", "geometry_key"],
            "row_level": "safe_exp_id",
        },
        "qc_filter": {"clean_qc_classes": sorted(CLEAN_QC_CLASSES),
                      "requires_xyz_exists": True},
        "feature_blocks": {
            "builtin": {
                "name": BUILTIN_BLOCK_NAME,
                "priority": 1,
                "columns": geom_status_report["builtin_scalar_features"],
                "source": "nearest-coreCN QC scalars off the accepted geometry",
            },
            "external": block_manifest,
            "structured_assets": (
                generated_artifacts["manifest"]["feature_blocks"]
                if generated_artifacts is not None else {}
            ),
            "planned_priority_order": [
                "1: complex scalar geometry/electronic (bond lengths, angles, CN, "
                "binding/strain energy, xTB charges, HOMO/LUMO/gap, dipole)",
                "2: explicit coordination-polyhedron distances/angles",
                "3: complex GFN2-xTB persistence image (image-native readout)",
                "4: simplicial / Vietoris-Rips inputs (LnSepNet)",
                "5: ligand-only persistence image (control block only)",
            ],
        },
        "feature_3d_columns": feat3d_cols,
        "status_columns": GEOMETRY_META_COLS,
        "weight_columns": weight_cols,
        "weights": weight_report,
        "geometry_status": geom_status_report,
        "row_counts": {
            "total_rows": int(len(final_3d)),
            "clean_geometry_rows": geom_status_report["clean_geometry_rows"],
            "rows_without_clean_geometry": geom_status_report["rows_without_clean_geometry"],
        },
        "guarantees": [
            "safe_exp_id kept as metadata/join key, excluded from is_feature_col()",
            "weight columns excluded from is_feature_col()",
            "rows with geometry_ok=False carry no 3D feature values (NaN)",
        ],
        "outputs": {
            "final_dataset_3d_parquet": str(FINAL_DATASET_3D_FILE) if parquet_3d_ok else None,
            "final_dataset_3d_preview_csv": str(FINAL_PREVIEW_3D_FILE),
            "sample_weights_csv": str(SAMPLE_WEIGHTS_FILE),
            "manifest_json": str(FEATURE_MANIFEST_FILE),
        },
        "unique_geometry_specs": geom_report.get("unique_geometry_specs"),
    }
    if generated_artifacts is not None:
        manifest["input_tables"]["generated_feature_manifest"] = str(
            generated_artifacts["manifest_path"]
        )
    with open(FEATURE_MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n=== 3D enrichment ===")
    print(json.dumps({
        "geometry_qc_tables_used": [Path(p).name for p in qc_files],
        "clean_geometry_rows": manifest["row_counts"]["clean_geometry_rows"],
        "rows_without_clean_geometry": manifest["row_counts"]["rows_without_clean_geometry"],
        "feature_3d_columns": len(feat3d_cols),
        "external_feature_blocks": [b["name"] for b in block_manifest],
        "weight_columns": weight_cols,
    }, indent=2))
    print(f"Wrote 3D dataset -> {FINAL_DATASET_3D_FILE}")
    print(f"Wrote weights    -> {SAMPLE_WEIGHTS_FILE}")
    print(f"Wrote manifest   -> {FEATURE_MANIFEST_FILE}")
    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="Directory of raw *.csv (default: raw_data/ or data/raw/).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N cleaned rows (smoke test).")
    ap.add_argument("--force-cn", type=int, default=None,
                    help="Force a fixed coreCN for all metals (default: metal-dependent CN(Z)).")
    ap.add_argument("--max-ligs", type=int, default=4, help="Cap on ligand copies per complex.")
    ap.add_argument("--with-3d-features", action="store_true",
                    help="Additionally write final_ml_dataset_3d.parquet joining clean "
                         "geometry-derived feature blocks. The no-3D baseline is unchanged.")
    ap.add_argument("--geometry-index", action="append", default=None,
                    help="Per-build_id geometry QC table(s) for --with-3d-features "
                         "(repeatable). Default: the reports/ QC queues + regenerated-accepted.")
    ap.add_argument("--feature-block", action="append", default=None,
                    help="External feature-block table as name:path[:join_key[:gating]] "
                         "(repeatable). join_key in build_id/geometry_key/safe_exp_id "
                         "(default build_id); gating in gated/ungated (default gated).")
    ap.add_argument(
        "--compute-geometry-features", action="store_true",
        help="With --with-3d-features, extract physical/polyhedron scalars, "
             "image-native complex + ligand-control PIs, and Vietoris-Rips inputs "
             "from existing QC-accepted XYZ files.",
    )
    ap.add_argument(
        "--feature-output-dir", type=Path, default=FEATURE_BLOCK_DIR,
        help="Output directory for computed geometry feature blocks/assets.",
    )
    ap.add_argument(
        "--pi-resolution", type=int, default=20,
        help="Persistence-image side length P (default: 20, matching PI400).",
    )
    ap.add_argument(
        "--vr-max-edge", type=float, default=4.0,
        help="Maximum Vietoris-Rips edge length in Angstrom (default: 4.0).",
    )
    args = ap.parse_args()

    if args.compute_geometry_features and not args.with_3d_features:
        ap.error("--compute-geometry-features requires --with-3d-features")
    if args.pi_resolution <= 1:
        ap.error("--pi-resolution must be greater than 1")
    if args.vr_max_edge <= 0:
        ap.error("--vr-max-edge must be positive")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    raw_dir = find_raw_dir(args.raw_dir)
    print(f"Reading raw data from {raw_dir} ...")
    raw = read_raw(raw_dir)

    print("Cleaning + deduplicating ...")
    clean, dedup_mapping, counts = clean_and_dedup(raw, limit=args.limit)
    if clean.empty:
        raise SystemExit("No usable rows after lanthanide/D/SMILES filters; check raw input or --limit.")
    print(f"  raw={counts['raw_rows']}  clean+dedup={counts['deduplicated_rows']}  "
          f"(exact-dup {counts['exact_duplicates_removed']}, "
          f"content-dup {counts['content_duplicates_removed']}, "
          f"invalid-smiles {counts['invalid_smiles_removed']})")

    print("Building ligand features (RDKit descriptors + ECFP) ...")
    clean, ligand_report = build_ligand_features(clean)

    print("Adding metal + condition features ...")
    clean, feat_report = add_metal_and_condition_features(clean)

    print("Planning geometry specs ...")
    clean, specs, geom_mapping, geom_report = build_geometry_specs(
        clean, force_cn=args.force_cn, max_ligs=args.max_ligs)
    if args.compute_geometry_features:
        print("Freezing existing stage-2 geometry templates from the merged index ...")
        clean, specs, geom_mapping, geom_report = freeze_plan_to_existing_geometry_index(
            clean,
            specs,
            geom_mapping,
            geom_report,
            PROCESSED_DIR / "geometry_index_merged.csv",
        )

    print("Dropping leakage/provenance + selecting final columns ...")
    final, column_report = select_final_columns(clean)

    # --- Write outputs ---
    try:
        final.to_parquet(FINAL_DATASET_FILE, index=False)
        parquet_ok = True
    except Exception as exc:  # pragma: no cover
        print(f"  WARNING: parquet write failed ({exc}); CSV preview still written.")
        parquet_ok = False
    final.head(200).to_csv(FINAL_PREVIEW_FILE, index=False)
    specs.to_csv(GEOMETRY_SPECS_FILE, index=False)
    geom_mapping.to_csv(GEOMETRY_DEDUP_MAPPING_FILE, index=False)
    geom_mapping.to_csv(GEOMETRY_CONDITION_MAPPING_FILE, index=False)
    dedup_mapping.to_csv(DEDUP_MAPPING_FILE, index=False)
    invalid_specs = specs[specs["geometry_status"] != "planned"]
    if len(invalid_specs):
        invalid_specs.to_csv(INVALID_SPECS_FILE, index=False)
    else:
        INVALID_SPECS_FILE.unlink(missing_ok=True)
    with open(DROPPED_COLUMNS_FILE, "w", encoding="utf-8") as fh:
        json.dump(column_report["dropped_columns"], fh, indent=2)

    info = {
        "raw_row_count": counts["raw_rows"],
        "after_lanthanide_filter": counts["after_lanthanide_filter"],
        "after_valid_D": counts["after_valid_D"],
        "after_smiles_present": counts["after_smiles_present"],
        "after_limit": counts["after_limit"],
        "limit": args.limit,
        "after_valid_canonical_smiles": counts["after_valid_canonical_smiles"],
        "cleaned_row_count": counts["after_valid_canonical_smiles"],
        "deduplicated_row_count": counts["deduplicated_rows"],
        "exact_duplicates_removed": counts["exact_duplicates_removed"],
        "content_duplicates_removed": counts["content_duplicates_removed"],
        "invalid_smiles_removed": counts["invalid_smiles_removed"],
        "unique_canonical_ligands": ligand_report["unique_canonical_ligands"],
        "unique_geometry_specs": geom_report["unique_geometry_specs"],
        "planned_geometry_specs": geom_report["planned_specs"],
        "invalid_no_donor_specs": geom_report["invalid_no_donor_specs"],
        "coreCN_distribution": geom_report["coreCN_distribution"],
        "inner_sphere_anion_distribution": geom_report["inner_sphere_anion_distribution"],
        "geometry_statuses": specs["geometry_status"].value_counts().to_dict(),
        "rows_with_xyz": 0,
        "rows_missing_geometry": int(len(final)),
        "geometry_note": "3D geometries are produced by build_unique_geometries.py (stage 2).",
        "feature_column_count": column_report["feature_columns"],
        "dataset_column_count": int(final.shape[1]),
        "dropped_column_count": column_report["dropped_column_count"],
        "dedup_key_columns": counts["content_dedup_key_columns"],
        "dedup_key_sources": counts["content_dedup_key_sources"],
        "content_dedup_excluded_provenance": counts["content_dedup_excluded_provenance"],
        "detected_columns": counts["detected_columns"],
        "condition_report": feat_report["condition_report"],
        "geometry_report": geom_report,
        "outputs": {
            "final_dataset_parquet": str(FINAL_DATASET_FILE) if parquet_ok else None,
            "final_dataset_preview_csv": str(FINAL_PREVIEW_FILE),
            "geometry_specs_csv": str(GEOMETRY_SPECS_FILE),
            "geometry_dedup_mapping_csv": str(GEOMETRY_DEDUP_MAPPING_FILE),
            "geometry_condition_mapping_csv": str(GEOMETRY_CONDITION_MAPPING_FILE),
            "dedup_mapping_csv": str(DEDUP_MAPPING_FILE),
            "invalid_smiles_csv": str(INVALID_SMILES_FILE) if counts["invalid_smiles_removed"] else None,
            "dropped_columns_json": str(DROPPED_COLUMNS_FILE),
            "info_json": str(INFO_FILE),
        },
    }
    with open(INFO_FILE, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

    print("\n=== summary ===")
    print(json.dumps({
        "raw_rows": info["raw_row_count"],
        "deduplicated_rows": info["deduplicated_row_count"],
        "unique_canonical_ligands": info["unique_canonical_ligands"],
        "unique_geometry_specs": info["unique_geometry_specs"],
        "coreCN_distribution": info["coreCN_distribution"],
        "inner_sphere_anion_distribution": info["inner_sphere_anion_distribution"],
        "feature_columns": info["feature_column_count"],
        "dataset_columns": info["dataset_column_count"],
    }, indent=2))
    print(f"\nWrote dataset -> {FINAL_DATASET_FILE}")
    print(f"Wrote specs   -> {GEOMETRY_SPECS_FILE}")
    print(f"Wrote info    -> {INFO_FILE}")

    # Optional stage 3: join clean 3D-derived features into a separate dataset.
    # The baseline outputs above are already written and are never modified here.
    if args.with_3d_features:
        build_and_write_3d_dataset(clean, args, geom_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
