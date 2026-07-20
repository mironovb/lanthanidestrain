"""Feature extraction from QC-accepted lanthanide complex geometries.

The stage-2 builder writes extended XYZ files.  Besides coordinates, those
files usually retain the xTB total energy, dipole and per-atom partial charges.
This module turns that existing information into auditable feature blocks and
keeps calculations that require new reference jobs (binding/strain energies and
frontier orbitals) explicitly missing instead of inventing values.

Persistence images are stored as P x P tensors for an image-native CNN/ViT
readout.  They are deliberately not flattened into the tabular dataset.
Vietoris--Rips inputs are stored as concatenated ragged arrays with pointer
arrays, ready for a simplicial model without recomputing the filtration.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LANTHANIDE_SYMBOLS = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}
DONOR_SYMBOLS = {"O", "N", "S", "P", "F", "Cl", "Br", "I"}

# Complete enough for every element that can occur in the current complexes.
_PERIODIC_SYMBOLS = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn",
)
ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate(_PERIODIC_SYMBOLS) if symbol}

MAX_COORDINATION_NUMBER = 9
PI_RESOLUTION = 20
PI_SPREAD = 0.08
PI_BIRTH_RANGE = (0.0, 2.5)
PI_DEATH_RANGE = (0.0, 2.5)
PI_HOMOLOGY_DIMS = (0, 1)
PI_RANDOM_SEED = 42
PI_MAX_EMBED_ATTEMPTS = 20
DEFAULT_VR_MAX_EDGE_ANGSTROM = 4.0

UNCOMPUTED_XTB_FEATURES = (
    "binding_energy_eV",
    "strain_energy_eV",
    "homo_eV",
    "lumo_eV",
    "homo_lumo_gap_eV",
)


@dataclass(frozen=True)
class ExtXYZGeometry:
    symbols: np.ndarray
    coordinates: np.ndarray
    partial_charges: np.ndarray
    energy_eV: float
    free_energy_eV: float
    dipole: np.ndarray
    comment: str

    @property
    def has_xtb_properties(self) -> bool:
        return bool(
            np.isfinite(self.energy_eV)
            and np.isfinite(self.dipole).all()
            and np.isfinite(self.partial_charges).all()
        )


def _float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _comment_fields(comment: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    try:
        tokens = shlex.split(comment)
    except ValueError:
        tokens = comment.split()
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def _property_layout(spec: str) -> dict[str, tuple[int, int]]:
    """Return extended-XYZ property name -> (start column, width)."""
    parts = str(spec).split(":")
    if len(parts) % 3:
        return {}
    layout: dict[str, tuple[int, int]] = {}
    offset = 0
    for idx in range(0, len(parts), 3):
        name = parts[idx]
        try:
            width = int(parts[idx + 2])
        except ValueError:
            return {}
        layout[name] = (offset, width)
        offset += width
    return layout


def read_extxyz(path: Path) -> ExtXYZGeometry:
    """Parse coordinates and saved xTB properties from an extended XYZ file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("xyz_has_too_few_lines")
    n_atoms = int(lines[0].strip())
    if len(lines) < n_atoms + 2:
        raise ValueError("xyz_atom_count_mismatch")

    comment = lines[1]
    fields = _comment_fields(comment)
    layout = _property_layout(fields.get("Properties", ""))
    charge_layout = layout.get("charge")

    symbols: list[str] = []
    coordinates: list[list[float]] = []
    charges: list[float] = []
    for atom_index, line in enumerate(lines[2:2 + n_atoms]):
        values = line.split()
        if len(values) < 4:
            raise ValueError(f"xyz_coordinate_line_{atom_index}_has_too_few_fields")
        symbols.append(values[0])
        coordinates.append([float(values[1]), float(values[2]), float(values[3])])
        charge = math.nan
        if charge_layout is not None:
            start, width = charge_layout
            # Property offsets include the species field at column zero.
            if width == 1 and start < len(values):
                charge = _float(values[start])
        charges.append(charge)

    dipole_text = fields.get("dipole", "")
    dipole_values = [_float(v) for v in dipole_text.split()]
    if len(dipole_values) != 3:
        dipole_values = [math.nan, math.nan, math.nan]

    return ExtXYZGeometry(
        symbols=np.asarray(symbols, dtype="U3"),
        coordinates=np.asarray(coordinates, dtype=np.float64),
        partial_charges=np.asarray(charges, dtype=np.float64),
        energy_eV=_float(fields.get("energy")),
        free_energy_eV=_float(fields.get("free_energy")),
        dipole=np.asarray(dipole_values, dtype=np.float64),
        comment=comment,
    )


def resolve_local_geometry_path(value: Any, repo_root: Path) -> Path | None:
    """Resolve local paths when a report still contains a cluster checkout path."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    direct = Path(text)
    if direct.exists():
        return direct.resolve()

    marker = f"{repo_root.name}/"
    if marker in text:
        local = repo_root / text.split(marker, 1)[1]
        if local.exists():
            return local.resolve()

    if not direct.is_absolute():
        local = repo_root / direct
        if local.exists():
            return local.resolve()
    return None


def _coordination_shell(
    geometry: ExtXYZGeometry,
    metal_symbol: str,
    core_cn: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    metal_indices = np.flatnonzero(geometry.symbols == str(metal_symbol))
    if metal_indices.size == 0:
        metal_indices = np.asarray(
            [i for i, symbol in enumerate(geometry.symbols) if symbol in LANTHANIDE_SYMBOLS],
            dtype=np.int64,
        )
    if metal_indices.size == 0:
        raise ValueError("lanthanide_metal_absent_from_xyz")
    metal_index = int(metal_indices[0])

    donor_indices = np.asarray(
        [i for i, symbol in enumerate(geometry.symbols) if symbol in DONOR_SYMBOLS],
        dtype=np.int64,
    )
    if donor_indices.size < core_cn:
        raise ValueError(f"only_{donor_indices.size}_donors_for_coreCN_{core_cn}")
    vectors = geometry.coordinates[donor_indices] - geometry.coordinates[metal_index]
    distances = np.linalg.norm(vectors, axis=1)
    order = np.argsort(distances, kind="stable")[:core_cn]
    return metal_index, donor_indices[order], vectors[order], distances[order]


def _angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 0:
        return math.nan
    cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def geometry_scalar_rows(
    build_id: str,
    geometry: ExtXYZGeometry,
    metal_symbol: str,
    core_cn: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Return physical scalars, explicit polyhedron scalars and donor indices."""
    metal_index, donor_indices, donor_vectors, donor_distances = _coordination_shell(
        geometry, metal_symbol, core_cn
    )
    donor_charges = geometry.partial_charges[donor_indices]

    physical: dict[str, Any] = {
        "build_id": build_id,
        "coordination_number": int(core_cn),
        "observed_donors_within_3p10A": int(
            sum(
                np.linalg.norm(geometry.coordinates[i] - geometry.coordinates[metal_index]) <= 3.10
                for i, symbol in enumerate(geometry.symbols)
                if symbol in DONOR_SYMBOLS
            )
        ),
        "complex_total_energy_eV": geometry.energy_eV,
        "complex_free_energy_eV": geometry.free_energy_eV,
        "dipole_x": float(geometry.dipole[0]),
        "dipole_y": float(geometry.dipole[1]),
        "dipole_z": float(geometry.dipole[2]),
        "dipole_magnitude": (
            float(np.linalg.norm(geometry.dipole))
            if np.isfinite(geometry.dipole).all() else math.nan
        ),
        "metal_partial_charge": float(geometry.partial_charges[metal_index]),
        **_summary(donor_distances, "ln_donor_distance"),
        **_summary(donor_charges, "donor_partial_charge"),
    }
    # These require new reference/single-point calculations.  Null is part of
    # the explicit schema; the companion queue records how they will be filled.
    physical.update({name: math.nan for name in UNCOMPUTED_XTB_FEATURES})

    polyhedron: dict[str, Any] = {
        "build_id": build_id,
        "coordination_number": int(core_cn),
    }
    for rank in range(MAX_COORDINATION_NUMBER):
        suffix = f"{rank + 1:02d}"
        if rank < len(donor_indices):
            atom_index = int(donor_indices[rank])
            polyhedron[f"ln_donor_distance_{suffix}"] = float(donor_distances[rank])
            polyhedron[f"donor_atomic_number_{suffix}"] = int(
                ATOMIC_NUMBER.get(str(geometry.symbols[atom_index]), 0)
            )
            physical[f"donor_partial_charge_{suffix}"] = float(donor_charges[rank])
        else:
            polyhedron[f"ln_donor_distance_{suffix}"] = math.nan
            polyhedron[f"donor_atomic_number_{suffix}"] = math.nan
            physical[f"donor_partial_charge_{suffix}"] = math.nan

    for left in range(MAX_COORDINATION_NUMBER):
        for right in range(left + 1, MAX_COORDINATION_NUMBER):
            name = f"donor_angle_{left + 1:02d}_{right + 1:02d}_deg"
            if left < len(donor_vectors) and right < len(donor_vectors):
                polyhedron[name] = _angle_degrees(donor_vectors[left], donor_vectors[right])
            else:
                polyhedron[name] = math.nan
    return physical, polyhedron, donor_indices


def persistence_diagram(coordinates: np.ndarray) -> np.ndarray:
    """Alpha-complex persistence diagram matching the existing PI400 benchmark."""
    try:
        import gudhi as gd
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("GUDHI is required for persistence images") from exc
    if coordinates.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    simplex_tree = gd.AlphaComplex(points=coordinates).create_simplex_tree()
    points: list[tuple[float, float]] = []
    for dim, interval in simplex_tree.persistence():
        if dim not in PI_HOMOLOGY_DIMS:
            continue
        birth, death = interval
        if np.isfinite(birth) and np.isfinite(death) and death > birth:
            points.append((float(birth), float(death)))
    return np.asarray(points, dtype=np.float64) if points else np.zeros((0, 2), dtype=np.float64)


def persistence_image(diagram: np.ndarray, resolution: int = PI_RESOLUTION) -> np.ndarray:
    """Render one full birth/death persistence image without flattening it."""
    birth_grid = np.linspace(PI_BIRTH_RANGE[0], PI_BIRTH_RANGE[1], resolution)
    death_grid = np.linspace(PI_DEATH_RANGE[0], PI_DEATH_RANGE[1], resolution)
    birth_mesh, death_mesh = np.meshgrid(birth_grid, death_grid, indexing="xy")
    image = np.zeros((resolution, resolution), dtype=np.float64)
    sigma2 = PI_SPREAD ** 2
    for birth, death in diagram:
        if not (PI_BIRTH_RANGE[0] <= birth <= PI_BIRTH_RANGE[1]):
            continue
        if not (PI_DEATH_RANGE[0] <= death <= PI_DEATH_RANGE[1]) or death <= birth:
            continue
        weight = death - birth
        exponent = -(
            (birth_mesh - birth) ** 2 + (death_mesh - death) ** 2
        ) / (2.0 * sigma2)
        image += weight * np.exp(exponent)
    return image.astype(np.float32)


def _ligand_coordinates(smiles: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid_smiles")
    mol = Chem.AddHs(mol)
    status = -1
    for attempt in range(PI_MAX_EMBED_ATTEMPTS):
        params = AllChem.ETKDGv3()
        params.randomSeed = PI_RANDOM_SEED + attempt
        params.useRandomCoords = attempt > 0
        status = AllChem.EmbedMolecule(mol, params)
        if status == 0:
            break
    if status != 0:
        raise ValueError("ligand_conformer_failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    conformer = mol.GetConformer()
    return np.asarray(
        [
            [
                conformer.GetAtomPosition(i).x,
                conformer.GetAtomPosition(i).y,
                conformer.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=np.float64,
    )


def _rips_simplices(coordinates: np.ndarray, max_edge: float):
    try:
        import gudhi as gd
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("GUDHI is required for Vietoris-Rips inputs") from exc
    tree = gd.RipsComplex(
        points=coordinates, max_edge_length=float(max_edge)
    ).create_simplex_tree(max_dimension=2)
    edges: list[list[int]] = []
    edge_filtration: list[float] = []
    triangles: list[list[int]] = []
    triangle_filtration: list[float] = []
    for simplex, filtration in tree.get_filtration():
        if len(simplex) == 2:
            edges.append([int(simplex[0]), int(simplex[1])])
            edge_filtration.append(float(filtration))
        elif len(simplex) == 3:
            triangles.append([int(simplex[0]), int(simplex[1]), int(simplex[2])])
            triangle_filtration.append(float(filtration))
    return (
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        np.asarray(edge_filtration, dtype=np.float32),
        np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
        np.asarray(triangle_filtration, dtype=np.float32),
    )


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_npz_atomic(path: Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _source_label(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def build_geometry_feature_artifacts(
    qc_index: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    repo_root: Path,
    output_dir: Path,
    pi_resolution: int = PI_RESOLUTION,
    vr_max_edge: float = DEFAULT_VR_MAX_EDGE_ANGSTROM,
) -> dict[str, Any]:
    """Compute all locally available 3D blocks and write ML-ready artifacts.

    ``rows`` is the deduplicated experiment table and must contain unique
    ``safe_exp_id`` values plus build_id/canonical_smiles.  Geometry calculations
    are deduplicated by build_id, then mapped back to safe_exp_id.
    """
    required = {"safe_exp_id", "build_id", "canonical_smiles"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"row table missing required columns: {missing}")
    safe_ids = rows["safe_exp_id"].astype("string")
    if safe_ids.isna().any() or (safe_ids.str.strip() == "").any():
        raise ValueError("safe_exp_id contains missing/empty values")
    if safe_ids.duplicated().any():
        examples = safe_ids[safe_ids.duplicated(False)].head(5).tolist()
        raise ValueError(f"safe_exp_id is not unique; examples={examples}")

    output_dir.mkdir(parents=True, exist_ok=True)
    qc = qc_index.copy()
    qc["build_id"] = qc["build_id"].astype("string")
    clean = qc[
        qc.get("qc_class", pd.Series(index=qc.index, dtype="string")).astype("string").eq("OK")
    ].drop_duplicates("build_id", keep="first")

    physical_rows: list[dict[str, Any]] = []
    polyhedron_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    geometries: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for ordinal, (_, row) in enumerate(clean.iterrows(), start=1):
        build_id = str(row["build_id"])
        geometry_key = str(row.get("geometry_key", "")).strip()
        if not geometry_key or geometry_key.lower() in {"nan", "none", "<na>"}:
            failure_rows.append({"build_id": build_id, "reason": "geometry_key_missing"})
            continue
        xyz_path = resolve_local_geometry_path(row.get("xyz_path"), repo_root)
        if xyz_path is None:
            failure_rows.append({"build_id": build_id, "reason": "local_xyz_not_found"})
            continue
        try:
            geometry = read_extxyz(xyz_path)
            core_cn = int(float(row.get("coreCN")))
            metal_symbol = str(row.get("metal_symbol", "")).strip()
            physical, polyhedron, donor_indices = geometry_scalar_rows(
                build_id, geometry, metal_symbol, core_cn
            )
        except Exception as exc:
            failure_rows.append({
                "build_id": build_id,
                "xyz_path": str(xyz_path),
                "reason": f"{type(exc).__name__}:{str(exc)[:240]}",
            })
            continue

        full_method = str(row.get("full_method", "")).strip()
        if full_method.lower() in {"nan", "none"}:
            full_method = ""
        relaxed = _bool_or_none(row.get("relax"))
        explicitly_non_xtb = bool(full_method and "xtb" not in full_method.lower()) or relaxed is False
        if explicitly_non_xtb:
            xtb_provenance = "excluded_non_xtb_method"
            pi_eligible = False
        elif full_method:
            xtb_provenance = f"confirmed_report:{full_method}"
            pi_eligible = geometry.has_xtb_properties
        elif geometry.has_xtb_properties:
            xtb_provenance = "extxyz_xtb_properties_present"
            pi_eligible = True
        else:
            xtb_provenance = "unverified_missing_xtb_properties"
            pi_eligible = False

        physical.pop("build_id", None)
        polyhedron.pop("build_id", None)
        physical_rows.append({"geometry_key": geometry_key, **physical})
        polyhedron_rows.append({"geometry_key": geometry_key, **polyhedron})
        status_rows.append({
            "build_id": build_id,
            "geometry_key": geometry_key,
            "geometry_source": str(row.get("geometry_source", "")),
            "xyz_path": str(xyz_path),
            "n_atoms": int(len(geometry.symbols)),
            "xtb_properties_available": bool(geometry.has_xtb_properties),
            "xtb_provenance": xtb_provenance,
            "complex_pi_eligible": bool(pi_eligible),
        })
        geometries.append({
            "build_id": build_id,
            "geometry_key": geometry_key,
            "geometry": geometry,
            "donor_indices": donor_indices,
            "pi_eligible": pi_eligible,
        })
        if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(clean):
            print(f"  3D feature extraction {ordinal}/{len(clean)}")

    physical_df = pd.DataFrame(physical_rows)
    polyhedron_df = pd.DataFrame(polyhedron_rows)
    status_df = pd.DataFrame(status_rows)
    failures_df = pd.DataFrame(failure_rows)

    physical_path = output_dir / "complex_physical_scalars.parquet"
    polyhedron_path = output_dir / "coordination_polyhedron.parquet"
    status_path = output_dir / "geometry_feature_status.csv"
    failures_path = output_dir / "geometry_feature_failures.csv"
    _write_parquet_atomic(physical_df, physical_path)
    _write_parquet_atomic(polyhedron_df, polyhedron_path)
    _write_csv_atomic(status_df, status_path)
    if not failures_df.empty:
        _write_csv_atomic(failures_df, failures_path)
    else:
        failures_path.unlink(missing_ok=True)

    # Complex PI: only structures with xTB provenance/properties are admitted.
    complex_pi_images: list[np.ndarray] = []
    complex_pi_build_ids: list[str] = []
    for item in geometries:
        if not item["pi_eligible"]:
            continue
        geometry = item["geometry"]
        image = persistence_image(
            persistence_diagram(geometry.coordinates), resolution=pi_resolution
        )
        complex_pi_images.append(image[np.newaxis, :, :])
        complex_pi_build_ids.append(item["build_id"])
    complex_pi_array = (
        np.stack(complex_pi_images).astype(np.float32)
        if complex_pi_images else np.zeros((0, 1, pi_resolution, pi_resolution), dtype=np.float32)
    )
    complex_pi_path = output_dir / "complex_gfn2xtb_pi_images.npz"
    _write_npz_atomic(
        complex_pi_path,
        images=complex_pi_array,
        build_ids=np.asarray(complex_pi_build_ids, dtype="U32"),
    )
    complex_pi_lookup = {
        item["geometry_key"]: idx
        for idx, item in enumerate(item for item in geometries if item["pi_eligible"])
    }

    # Ligand-only PI remains a control asset and is never joined as a feature.
    ligand_pi_images: list[np.ndarray] = []
    ligand_smiles: list[str] = []
    ligand_failures: list[dict[str, str]] = []
    for smiles in sorted(rows["canonical_smiles"].dropna().astype(str).unique()):
        try:
            coordinates = _ligand_coordinates(smiles)
            image = persistence_image(persistence_diagram(coordinates), resolution=pi_resolution)
        except Exception as exc:
            ligand_failures.append({
                "canonical_smiles": smiles,
                "reason": f"{type(exc).__name__}:{str(exc)[:180]}",
            })
            continue
        ligand_smiles.append(smiles)
        ligand_pi_images.append(image[np.newaxis, :, :])
    ligand_pi_array = (
        np.stack(ligand_pi_images).astype(np.float32)
        if ligand_pi_images else np.zeros((0, 1, pi_resolution, pi_resolution), dtype=np.float32)
    )
    ligand_pi_path = output_dir / "ligand_pi_control_images.npz"
    _write_npz_atomic(
        ligand_pi_path,
        images=ligand_pi_array,
        canonical_smiles=np.asarray(ligand_smiles, dtype="U1024"),
    )
    ligand_lookup = {smiles: idx for idx, smiles in enumerate(ligand_smiles)}
    ligand_failure_path = output_dir / "ligand_pi_control_failures.csv"
    if ligand_failures:
        _write_csv_atomic(pd.DataFrame(ligand_failures), ligand_failure_path)
    else:
        ligand_failure_path.unlink(missing_ok=True)

    # Vietoris--Rips ragged arrays for LnSepNet / simplicial models.
    node_coordinates: list[np.ndarray] = []
    node_atomic_numbers: list[np.ndarray] = []
    node_partial_charges: list[np.ndarray] = []
    node_is_metal: list[np.ndarray] = []
    node_is_coord_donor: list[np.ndarray] = []
    edge_indices: list[np.ndarray] = []
    edge_filtration: list[np.ndarray] = []
    triangle_indices: list[np.ndarray] = []
    triangle_filtration: list[np.ndarray] = []
    node_ptr = [0]
    edge_ptr = [0]
    triangle_ptr = [0]
    vr_build_ids: list[str] = []

    for item in geometries:
        geometry = item["geometry"]
        edges, edge_f, triangles, triangle_f = _rips_simplices(
            geometry.coordinates, vr_max_edge
        )
        node_offset = node_ptr[-1]
        node_coordinates.append(geometry.coordinates.astype(np.float32))
        node_atomic_numbers.append(np.asarray(
            [ATOMIC_NUMBER.get(str(symbol), 0) for symbol in geometry.symbols], dtype=np.int16
        ))
        node_partial_charges.append(geometry.partial_charges.astype(np.float32))
        node_is_metal.append(np.asarray(
            [symbol in LANTHANIDE_SYMBOLS for symbol in geometry.symbols], dtype=np.int8
        ))
        donor_mask = np.zeros(len(geometry.symbols), dtype=np.int8)
        donor_mask[item["donor_indices"]] = 1
        node_is_coord_donor.append(donor_mask)
        edge_indices.append(edges + node_offset)
        edge_filtration.append(edge_f)
        triangle_indices.append(triangles + node_offset)
        triangle_filtration.append(triangle_f)
        node_ptr.append(node_ptr[-1] + len(geometry.symbols))
        edge_ptr.append(edge_ptr[-1] + len(edges))
        triangle_ptr.append(triangle_ptr[-1] + len(triangles))
        vr_build_ids.append(item["build_id"])

    def concatenate(arrays: list[np.ndarray], shape: tuple[int, ...], dtype):
        return np.concatenate(arrays, axis=0) if arrays else np.zeros(shape, dtype=dtype)

    vr_path = output_dir / "vietoris_rips_inputs.npz"
    _write_npz_atomic(
        vr_path,
        coordinates=concatenate(node_coordinates, (0, 3), np.float32),
        atomic_numbers=concatenate(node_atomic_numbers, (0,), np.int16),
        partial_charges=concatenate(node_partial_charges, (0,), np.float32),
        is_metal=concatenate(node_is_metal, (0,), np.int8),
        is_coord_donor=concatenate(node_is_coord_donor, (0,), np.int8),
        node_ptr=np.asarray(node_ptr, dtype=np.int64),
        edge_index=concatenate(edge_indices, (0, 2), np.int64).T,
        edge_filtration=concatenate(edge_filtration, (0,), np.float32),
        edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
        triangle_index=concatenate(triangle_indices, (0, 3), np.int64).T,
        triangle_filtration=concatenate(triangle_filtration, (0,), np.float32),
        triangle_ptr=np.asarray(triangle_ptr, dtype=np.int64),
        build_ids=np.asarray(vr_build_ids, dtype="U32"),
    )
    vr_lookup = {item["geometry_key"]: idx for idx, item in enumerate(geometries)}

    row_index = rows[["safe_exp_id", "build_id", "geometry_key", "canonical_smiles"]].copy()
    feature_build_lookup = status_df.set_index("geometry_key")["build_id"]
    row_index["geometry_feature_build_id"] = (
        row_index["geometry_key"].astype("string").map(feature_build_lookup).astype("string")
    )
    row_index["complex_pi_image_index"] = (
        row_index["geometry_key"].astype(str).map(complex_pi_lookup).astype("Int64")
    )
    row_index["vr_graph_index"] = (
        row_index["geometry_key"].astype(str).map(vr_lookup).astype("Int64")
    )
    row_index["ligand_pi_control_image_index"] = (
        row_index["canonical_smiles"].astype(str).map(ligand_lookup).astype("Int64")
    )
    row_index_path = output_dir / "row_asset_index.parquet"
    _write_parquet_atomic(row_index, row_index_path)

    # Explicit safe_exp_id-keyed feature block requested by the dataset contract.
    by_safe = rows[["safe_exp_id", "build_id", "geometry_key"]].merge(
        physical_df, on="geometry_key", how="left", validate="many_to_one"
    ).merge(
        polyhedron_df, on="geometry_key", how="left", validate="many_to_one",
        suffixes=("__physical", "__polyhedron"),
    )
    by_safe = by_safe.merge(
        row_index.drop(columns=["canonical_smiles"]),
        on=["safe_exp_id", "build_id", "geometry_key"], how="left", validate="one_to_one",
    )
    by_safe_path = output_dir / "features_by_safe_exp_id.parquet"
    _write_parquet_atomic(by_safe, by_safe_path)

    reference_queue = clean[[
        col for col in [
            "build_id", "metal_symbol", "metal_ox", "SMILES_FOR_ARCHITECTOR",
            "coreCN", "n_ligs", "inner_sphere_anion", "fill_ligand", "n_fill",
            "xyz_path",
        ] if col in clean.columns
    ]].copy()
    reference_queue["required_outputs"] = (
        "binding_energy_eV;strain_energy_eV;homo_eV;lumo_eV;homo_lumo_gap_eV"
    )
    reference_queue["calculation_status"] = "not_run_requires_reference_xtb"
    reference_queue["binding_energy_definition"] = (
        "E_complex-(E_Ln_ion+n_ligs*E_free_ligand+n_fill*E_free_fill); "
        "all references must use the same xTB method/charge convention"
    )
    reference_queue_path = output_dir / "xtb_reference_calculation_queue.csv"
    _write_csv_atomic(reference_queue, reference_queue_path)

    manifest = {
        "description": "3D feature artifacts from QC-accepted existing complex geometries",
        "keys": {
            "row": "safe_exp_id",
            "scientific_geometry": "geometry_key",
            "accepted_geometry_provenance": "geometry_feature_build_id",
        },
        "row_count": int(len(rows)),
        "clean_geometry_count": int(len(clean)),
        "geometry_features_computed": int(len(physical_df)),
        "geometry_feature_failures": int(len(failures_df)),
        "feature_blocks": {
            "complex_physical_scalars": {
                "role": "model_feature",
                "priority": 1,
                "path": _source_label(physical_path, repo_root),
                "computed_columns": [
                    c for c in physical_df.columns
                    if c not in {"geometry_key", *UNCOMPUTED_XTB_FEATURES}
                ],
                "uncomputed_null_columns": list(UNCOMPUTED_XTB_FEATURES),
                "uncomputed_reason": (
                    "existing extxyz lacks free-ion/free-ligand reference energies and "
                    "individual frontier orbital energies"
                ),
                "reference_queue": _source_label(reference_queue_path, repo_root),
            },
            "coordination_polyhedron": {
                "role": "model_feature",
                "priority": 2,
                "path": _source_label(polyhedron_path, repo_root),
                "donor_order": "nearest-to-metal radial rank",
                "max_coordination_number": MAX_COORDINATION_NUMBER,
                "representation": "raw distances and donor-M-donor angles; no averaging",
            },
            "complex_gfn2xtb_persistence_image": {
                "role": "model_feature_image_native",
                "priority": 3,
                "path": _source_label(complex_pi_path, repo_root),
                "shape": list(complex_pi_array.shape),
                "per_sample_shape": [1, pi_resolution, pi_resolution],
                "readout": "CNN_or_ViT; do_not_flatten_into_tabular_MLP",
                "filtration": "AlphaComplex (same PI400 benchmark convention)",
                "homology_dimensions": list(PI_HOMOLOGY_DIMS),
                "birth_range": list(PI_BIRTH_RANGE),
                "death_range": list(PI_DEATH_RANGE),
                "spread": PI_SPREAD,
                "included_geometries": int(len(complex_pi_build_ids)),
            },
            "vietoris_rips": {
                "role": "simplicial_model_input",
                "priority": 4,
                "path": _source_label(vr_path, repo_root),
                "max_edge_length_angstrom": float(vr_max_edge),
                "max_simplex_dimension": 2,
                "geometries": len(vr_build_ids),
                "nodes": int(node_ptr[-1]),
                "edges": int(edge_ptr[-1]),
                "triangles": int(triangle_ptr[-1]),
            },
            "ligand_persistence_image_control": {
                "role": "control_only_not_model_feature",
                "priority": 5,
                "path": _source_label(ligand_pi_path, repo_root),
                "shape": list(ligand_pi_array.shape),
                "source": "RDKit ETKDGv3 ligand conformer",
                "failures": len(ligand_failures),
            },
        },
        "row_asset_index": _source_label(row_index_path, repo_root),
        "safe_exp_id_feature_table": _source_label(by_safe_path, repo_root),
        "status_table": _source_label(status_path, repo_root),
        "failure_table": _source_label(failures_path, repo_root) if not failures_df.empty else None,
        "scientific_guarantees": [
            "only qc_class=OK geometries enter complex-derived blocks",
            "complex PI is image-native and is not flattened into the tabular dataset",
            "ligand PI is stored as a control asset and is not joined as a model feature",
            "binding/strain/frontier-orbital values remain null until reference xTB jobs run",
        ],
    }
    manifest_path = output_dir / "feature_blocks_manifest.json"
    _write_json_atomic(manifest, manifest_path)

    return {
        "block_specs": [
            {
                "name": "complex_physical",
                "path": physical_path,
                "join_key": "geometry_key",
                "gated": True,
            },
            {
                "name": "polyhedron",
                "path": polyhedron_path,
                "join_key": "geometry_key",
                "gated": True,
            },
        ],
        "row_asset_columns": row_index[[
            "safe_exp_id", "complex_pi_image_index", "vr_graph_index",
            "ligand_pi_control_image_index",
        ]],
        "manifest": manifest,
        "manifest_path": manifest_path,
    }
