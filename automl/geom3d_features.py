#!/usr/bin/env python3
"""Rich 3D descriptors from GFN2-xTB optimised Architector complex geometries.

Scope
-----
This module *reads* extended-XYZ geometries that already exist on disk and turns
them into named, auditable descriptor blocks.  It never writes into ``data/``
and it never regenerates a geometry.  Everything is derived from the atoms,
coordinates, xTB partial charges, xTB forces and the energies stored in the
extxyz comment line.

Why these blocks
----------------
The baseline (2D) model already knows the ligand graph (ECFP + RDKit
descriptors), the metal identity (Z, ionic radius) and the experimental
conditions.  A leave-extractants-out split therefore has an easy job *between*
extractants (ligand identity is highly predictive of average log D) and a hard
job *within* an extractant, where the only thing that changes is the metal and
the conditions.  Any 3D block that only encodes "which ligand is this" is
redundant.  The blocks below are deliberately organised so that the
metal-sensitive, contraction-corrected quantities are separable from the purely
ligand-shaped ones:

  G1  first_shell     inner coordination sphere distances / donor composition
  G2  contraction     the same shell after subtracting the Shannon ionic radius
                      (removes the trivial lanthanide-contraction trend, leaves
                      the ligand-specific fit-to-metal signal)
  G3  polyhedron      shape of the donor polyhedron (CShM vs ideal polyhedra,
                      inertia/asphericity, convex-hull volume, solid angles)
  G4  steric          %V_bur, exposure of the metal, radial atom counts
  G5  electronic      xTB charges, charge transfer, dipole geometry, force
                      residuals (a strain proxy)
  G6  rdf             element-resolved metal-centred radial distribution
                      functions (smooth 3D fingerprint)
  G7  global_shape    whole-complex size/shape/lipophilic-surface descriptors
                      (organic-phase solvation proxies)
  G8  chelate         donor-donor connectivity, bite angles, chelate ring sizes
  G9  topology        persistence statistics (H0/H1) of the complex point cloud
                      and of the metal-centred neighbourhood

Each block is emitted with a ``g<N>__`` prefix so the AutoML driver can switch
blocks on and off and attribute R^2 gains to a specific physical hypothesis.

Usage
-----
    python -m automl.geom3d_features --shard 0 --num-shards 16 \
        --out automl/artifacts/geom3d_shard0.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay  # noqa: F401  (ConvexHull used)
from scipy.spatial.distance import pdist, squareform

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.geometry_features import read_extxyz, ExtXYZGeometry  # noqa: E402
from src.chemistry.coordination import LANTHANIDE_DESCRIPTORS  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LANTHANIDE_SYMBOLS = set(LANTHANIDE_DESCRIPTORS) | {"Pm", "Y"}
DONOR_SYMBOLS = ("O", "N", "S", "P", "F", "Cl", "Br", "I")
ELEMENT_GROUPS = ("H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I")

# Shannon effective ionic radii, Ln(3+).  Reused from the stage-1 chemistry
# table so the AutoML features can never disagree with the dataset builder.
IONIC_RADIUS = {k: v["Ionic Radius_metal"] for k, v in LANTHANIDE_DESCRIPTORS.items()}
IONIC_RADIUS["Pm"] = 1.093  # interpolated Nd/Sm; only used if a Pm row appears
IONIC_RADIUS["Y"] = 1.019

# Covalent radii (Cordero 2008) for connectivity perception.
COVALENT_RADIUS = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
    "Na": 1.66, "K": 2.03, "Ca": 1.76, "Mg": 1.41, "Se": 1.20,
}
DEFAULT_COVALENT_RADIUS = 1.50

# Bondi van der Waals radii for %V_bur / SASA.
VDW_RADIUS = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Se": 1.90,
}
DEFAULT_VDW_RADIUS = 2.00

# Pauling electronegativity, used for a simple polarity descriptor.
ELECTRONEGATIVITY = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "P": 2.19,
    "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66, "Si": 1.90, "Se": 2.55,
}

# Donor-shell cutoff.  3.10 A is the convention already used by
# ``src/geometry_features.py`` for "observed donors"; keep it identical so the
# new block is comparable with the existing one.
DONOR_CUTOFF_A = 3.10

RDF_GRID = np.arange(1.8, 8.01, 0.20)      # 32 bins, metal-centred
RDF_ETA = 12.0                              # Gaussian sharpness (1/A^2)
SHELL_EDGES = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


# ---------------------------------------------------------------------------
# Ideal reference polyhedra for continuous shape measures (CShM)
# ---------------------------------------------------------------------------
def _normalise_vertices(v: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale so that the mean radius is 1."""
    v = np.asarray(v, dtype=float)
    v = v - v.mean(axis=0)
    scale = np.sqrt((v ** 2).sum(axis=1).mean())
    return v / scale if scale > 0 else v


def _square_antiprism() -> np.ndarray:
    """D4d square antiprism, CN 8 (SAPR-8)."""
    top = [[np.cos(t), np.sin(t), 0.5] for t in np.arange(4) * np.pi / 2]
    bot = [[np.cos(t + np.pi / 4), np.sin(t + np.pi / 4), -0.5]
           for t in np.arange(4) * np.pi / 2]
    return _normalise_vertices(np.array(top + bot))


def _triangular_dodecahedron() -> np.ndarray:
    """D2d triangular dodecahedron / bisdisphenoid, CN 8 (TDD-8)."""
    a, b = 1.0, 0.6
    v = [
        [a, 0, b], [-a, 0, b], [0, a, -b], [0, -a, -b],
        [b, 0, -a], [-b, 0, -a], [0, b, a], [0, -b, a],
    ]
    return _normalise_vertices(np.array(v, dtype=float))


def _cube() -> np.ndarray:
    """Oh cube, CN 8 (CU-8)."""
    v = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    return _normalise_vertices(np.array(v, dtype=float))


def _hexagonal_bipyramid() -> np.ndarray:
    """D6h hexagonal bipyramid, CN 8 (HBPY-8)."""
    ring = [[np.cos(t), np.sin(t), 0.0] for t in np.arange(6) * np.pi / 3]
    return _normalise_vertices(np.array(ring + [[0, 0, 1.0], [0, 0, -1.0]]))


def _tricapped_trigonal_prism() -> np.ndarray:
    """D3h tricapped trigonal prism, CN 9 (TCTPR-9) -- the classic Ln(III) shape."""
    h = 0.75
    top = [[np.cos(t), np.sin(t), h] for t in np.arange(3) * 2 * np.pi / 3]
    bot = [[np.cos(t), np.sin(t), -h] for t in np.arange(3) * 2 * np.pi / 3]
    caps = [[1.45 * np.cos(t + np.pi / 3), 1.45 * np.sin(t + np.pi / 3), 0.0]
            for t in np.arange(3) * 2 * np.pi / 3]
    return _normalise_vertices(np.array(top + bot + caps))


def _capped_square_antiprism() -> np.ndarray:
    """C4v monocapped square antiprism, CN 9 (CSAPR-9)."""
    sap = _square_antiprism()
    return _normalise_vertices(np.vstack([sap, [0.0, 0.0, 1.35]]))


def _muffin() -> np.ndarray:
    """Cs muffin, CN 9 (MFF-9)."""
    ring5 = [[np.cos(t), np.sin(t), -0.45] for t in np.arange(5) * 2 * np.pi / 5]
    ring3 = [[0.9 * np.cos(t), 0.9 * np.sin(t), 0.6] for t in np.arange(3) * 2 * np.pi / 3]
    return _normalise_vertices(np.array(ring5 + ring3 + [[0.0, 0.0, 1.2]]))


def _heptagonal_bipyramid() -> np.ndarray:
    """D7h heptagonal bipyramid, CN 9 (HBPY-9)."""
    ring = [[np.cos(t), np.sin(t), 0.0] for t in np.arange(7) * 2 * np.pi / 7]
    return _normalise_vertices(np.array(ring + [[0, 0, 1.0], [0, 0, -1.0]]))


def _pentagonal_bipyramid() -> np.ndarray:
    """D5h pentagonal bipyramid, CN 7 (PBPY-7)."""
    ring = [[np.cos(t), np.sin(t), 0.0] for t in np.arange(5) * 2 * np.pi / 5]
    return _normalise_vertices(np.array(ring + [[0, 0, 1.1], [0, 0, -1.1]]))


def _capped_octahedron() -> np.ndarray:
    """C3v capped octahedron, CN 7 (COC-7)."""
    oct_v = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    return _normalise_vertices(np.array(oct_v + [[0.75, 0.75, 0.75]], dtype=float))


def _octahedron() -> np.ndarray:
    """Oh octahedron, CN 6 (OC-6)."""
    return _normalise_vertices(np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float))


def _trigonal_prism() -> np.ndarray:
    """D3h trigonal prism, CN 6 (TPR-6)."""
    top = [[np.cos(t), np.sin(t), 0.8] for t in np.arange(3) * 2 * np.pi / 3]
    bot = [[np.cos(t), np.sin(t), -0.8] for t in np.arange(3) * 2 * np.pi / 3]
    return _normalise_vertices(np.array(top + bot))


def _bicapped_square_antiprism() -> np.ndarray:
    """D4d bicapped square antiprism, CN 10 (BCSAPR-10)."""
    sap = _square_antiprism()
    return _normalise_vertices(np.vstack([sap, [0, 0, 1.35], [0, 0, -1.35]]))


def _sphenocorona() -> np.ndarray:
    """J87 sphenocorona-like CN 10 alternative (SPC-10, approximate)."""
    ring5a = [[np.cos(t), np.sin(t), -0.55] for t in np.arange(5) * 2 * np.pi / 5]
    ring5b = [[np.cos(t + np.pi / 5), np.sin(t + np.pi / 5), 0.55]
              for t in np.arange(5) * 2 * np.pi / 5]
    return _normalise_vertices(np.array(ring5a + ring5b))


REFERENCE_POLYHEDRA: dict[int, dict[str, np.ndarray]] = {
    6: {"OC": _octahedron(), "TPR": _trigonal_prism()},
    7: {"PBPY": _pentagonal_bipyramid(), "COC": _capped_octahedron()},
    8: {"SAPR": _square_antiprism(), "TDD": _triangular_dodecahedron(),
        "CU": _cube(), "HBPY": _hexagonal_bipyramid()},
    9: {"TCTPR": _tricapped_trigonal_prism(), "CSAPR": _capped_square_antiprism(),
        "MFF": _muffin(), "HBPY9": _heptagonal_bipyramid()},
    10: {"BCSAPR": _bicapped_square_antiprism(), "SPC": _sphenocorona()},
}
# Every shape name that can be produced, so the output table has a stable schema.
ALL_SHAPE_NAMES = sorted({n for d in REFERENCE_POLYHEDRA.values() for n in d})


# ---------------------------------------------------------------------------
# Small geometric helpers
# ---------------------------------------------------------------------------
def _kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """Optimal-rotation RMSD between two equally sized, centred point sets."""
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    diff = (rot @ p.T).T - q
    return float(np.sqrt((diff ** 2).sum() / len(p)))


def _greedy_assign(shell: np.ndarray, ref: np.ndarray, n_restarts: int = 24) -> float:
    """CShM-style minimal RMSD over vertex permutations.

    Exact CShM enumerates all N! vertex labellings.  For N up to 10 that is
    3.6M permutations per geometry, which is too slow for a full sweep, so we
    use a randomised greedy assignment with restarts: align on a random seed
    triple, then match remaining vertices by nearest neighbour.  This is an
    upper bound on the true CShM and is monotone in the same direction, which
    is all a learned feature needs.
    """
    n = len(shell)
    best = np.inf
    rng = np.random.default_rng(0xC0FFEE + n)
    for _ in range(n_restarts):
        perm = rng.permutation(n)
        # Seed with a random 3-point alignment, then refine by nearest-neighbour
        # matching under the resulting rotation (2 refinement sweeps).
        order = perm.copy()
        for _sweep in range(3):
            h = shell[order].T @ ref
            u, _, vt = np.linalg.svd(h)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
            rotated = (rot @ shell.T).T
            cost = ((rotated[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
            # Greedy nearest assignment (Hungarian would be exact but slower and
            # the difference is negligible for these near-ideal shells).
            taken_r, taken_c = set(), {}
            for i, j in zip(*np.unravel_index(np.argsort(cost, axis=None), cost.shape)):
                if i in taken_r or j in taken_c.values():
                    continue
                taken_r.add(int(i))
                taken_c[int(i)] = int(j)
                if len(taken_r) == n:
                    break
            new_order = np.empty(n, dtype=int)
            for i, j in taken_c.items():
                new_order[j] = i
            if np.array_equal(new_order, order):
                break
            order = new_order
        best = min(best, _kabsch_rmsd(shell[order], ref))
    return best


def continuous_shape_measures(vectors: np.ndarray) -> dict[str, float]:
    """CShM-like deviation of the donor shell from each ideal polyhedron.

    ``vectors`` are donor positions relative to the metal.  They are size
    normalised first so the measure is scale free (pure shape).  Returned value
    is ``100 * min_RMSD^2`` in the CShM spirit: 0 = ideal, larger = distorted.
    """
    out = {f"cshm_{name}": math.nan for name in ALL_SHAPE_NAMES}
    n = len(vectors)
    refs = REFERENCE_POLYHEDRA.get(n)
    out["cshm_n_vertices"] = float(n)
    if refs is None or n < 4:
        out["cshm_best"] = math.nan
        out["cshm_best_minus_second"] = math.nan
        return out
    shell = _normalise_vertices(vectors)
    scored: list[tuple[str, float]] = []
    for name, ref in refs.items():
        value = 100.0 * _greedy_assign(shell, ref) ** 2
        out[f"cshm_{name}"] = value
        scored.append((name, value))
    scored.sort(key=lambda kv: kv[1])
    out["cshm_best"] = scored[0][1]
    out["cshm_best_minus_second"] = (
        scored[1][1] - scored[0][1] if len(scored) > 1 else math.nan
    )
    return out


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Quasi-uniform points on the unit sphere (for SASA / %V_bur sampling)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


_SPHERE_LOW = _fibonacci_sphere(194)
_SPHERE_HIGH = _fibonacci_sphere(302)


def shrake_rupley(coords: np.ndarray, radii: np.ndarray,
                  probe: float = 1.40, points: np.ndarray | None = None
                  ) -> np.ndarray:
    """Per-atom solvent-accessible surface area (Shrake-Rupley)."""
    pts = _SPHERE_LOW if points is None else points
    n = len(coords)
    r = radii + probe
    areas = np.zeros(n)
    # Neighbour prefilter using a distance matrix (complexes are <= ~500 atoms).
    dmat = squareform(pdist(coords))
    for i in range(n):
        cand = np.flatnonzero((dmat[i] < (r[i] + r.max())) & (np.arange(n) != i))
        if cand.size == 0:
            areas[i] = 4.0 * np.pi * r[i] ** 2
            continue
        surf = coords[i] + r[i] * pts
        d = np.linalg.norm(surf[:, None, :] - coords[cand][None, :, :], axis=2)
        accessible = np.all(d >= r[cand][None, :], axis=1)
        areas[i] = 4.0 * np.pi * r[i] ** 2 * accessible.mean()
    return areas


def buried_volume(coords: np.ndarray, radii: np.ndarray, centre: np.ndarray,
                  sphere_radius: float, n_grid: int = 40) -> float:
    """Percent buried volume %V_bur inside a sphere around the metal.

    Standard steric descriptor from organometallic chemistry (Cavallo's
    SambVca).  Implemented on a regular cubic grid; the metal itself is
    excluded from the occupying atoms.
    """
    lin = np.linspace(-sphere_radius, sphere_radius, n_grid)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    inside = np.linalg.norm(grid, axis=1) <= sphere_radius
    grid = grid[inside] + centre
    if grid.size == 0:
        return math.nan
    occupied = np.zeros(len(grid), dtype=bool)
    # Chunk to keep memory bounded for large complexes.
    for start in range(0, len(coords), 64):
        block = coords[start:start + 64]
        block_r = radii[start:start + 64]
        d = np.linalg.norm(grid[:, None, :] - block[None, :, :], axis=2)
        occupied |= np.any(d <= block_r[None, :], axis=1)
    return float(100.0 * occupied.mean())


def solid_angle_fraction(vectors: np.ndarray, radii: np.ndarray) -> float:
    """Fraction of the 4-pi sphere at the metal shadowed by the ligand atoms."""
    d = np.linalg.norm(vectors, axis=1)
    keep = d > 1e-6
    if not np.any(keep):
        return math.nan
    u = vectors[keep] / d[keep, None]
    r = np.clip(radii[keep] / d[keep], 0.0, 0.999)
    half_angle = np.arcsin(r)
    cos_lim = np.cos(half_angle)
    covered = np.zeros(len(_SPHERE_HIGH), dtype=bool)
    for start in range(0, len(u), 128):
        block_u = u[start:start + 128]
        block_c = cos_lim[start:start + 128]
        covered |= np.any(_SPHERE_HIGH @ block_u.T >= block_c[None, :], axis=1)
    return float(covered.mean())


def _inertia_descriptors(points: np.ndarray, weights: np.ndarray | None = None
                         ) -> tuple[np.ndarray, float, float]:
    """Principal moments, asphericity and acylindricity of a point cloud."""
    w = np.ones(len(points)) if weights is None else np.asarray(weights, float)
    centre = (points * w[:, None]).sum(axis=0) / w.sum()
    rel = points - centre
    gyration = (w[:, None, None] * rel[:, :, None] * rel[:, None, :]).sum(axis=0) / w.sum()
    eig = np.sort(np.linalg.eigvalsh(gyration))  # l1 <= l2 <= l3
    asphericity = float(eig[2] - 0.5 * (eig[0] + eig[1]))
    acylindricity = float(eig[1] - eig[0])
    return eig, asphericity, acylindricity


# ---------------------------------------------------------------------------
# Descriptor blocks
# ---------------------------------------------------------------------------
def _metal_index(geom: ExtXYZGeometry, metal_symbol: str | None) -> int:
    if metal_symbol:
        hit = np.flatnonzero(geom.symbols == str(metal_symbol))
        if hit.size:
            return int(hit[0])
    hit = np.asarray([i for i, s in enumerate(geom.symbols) if s in LANTHANIDE_SYMBOLS])
    if hit.size == 0:
        raise ValueError("no_lanthanide_in_geometry")
    return int(hit[0])


def block_first_shell(geom: ExtXYZGeometry, mi: int) -> dict[str, float]:
    """G1: raw inner-sphere metrics with a distance cutoff (no CN assumption)."""
    vecs = geom.coordinates - geom.coordinates[mi]
    dist = np.linalg.norm(vecs, axis=1)
    dist[mi] = np.inf
    is_donor = np.isin(geom.symbols, DONOR_SYMBOLS)
    shell = np.flatnonzero(is_donor & (dist <= DONOR_CUTOFF_A))
    out: dict[str, float] = {}
    out["cn_observed"] = float(shell.size)
    if shell.size == 0:
        return out
    sd = np.sort(dist[shell])
    out["d_min"] = float(sd[0])
    out["d_max"] = float(sd[-1])
    out["d_mean"] = float(sd.mean())
    out["d_std"] = float(sd.std())
    out["d_range"] = float(sd[-1] - sd[0])
    out["d_median"] = float(np.median(sd))
    # Radial "tightness": inverse-cube weighted mean is what a crystal-field
    # splitting actually feels.
    out["d_inv3_sum"] = float((1.0 / sd ** 3).sum())
    out["d_inv6_sum"] = float((1.0 / sd ** 6).sum())
    # Gap to the next non-bonded atom -- how cleanly the shell is defined.
    outside = dist[np.isfinite(dist) & (dist > sd[-1])]
    out["shell_gap"] = float(outside.min() - sd[-1]) if outside.size else math.nan
    # Donor composition.
    for element in DONOR_SYMBOLS:
        out[f"n_donor_{element}"] = float(np.sum(geom.symbols[shell] == element))
    en = np.array([ELECTRONEGATIVITY.get(s, 2.5) for s in geom.symbols[shell]])
    out["donor_en_mean"] = float(en.mean())
    out["donor_en_std"] = float(en.std())
    out["donor_en_max"] = float(en.max())
    # Hard/soft split: O/F = hard, N = borderline, S/P/Cl/Br/I = soft.
    hard = np.isin(geom.symbols[shell], ("O", "F")).sum()
    soft = np.isin(geom.symbols[shell], ("S", "P", "Cl", "Br", "I")).sum()
    out["donor_hard_frac"] = float(hard / shell.size)
    out["donor_soft_frac"] = float(soft / shell.size)
    return out


def block_contraction(geom: ExtXYZGeometry, mi: int, metal_symbol: str
                      ) -> dict[str, float]:
    """G2: shell metrics with the Shannon ionic radius removed.

    ``d(M-L) - r_ionic(M)`` is the part of the bond length that belongs to the
    *ligand*, not to the metal size.  Within one extractant this is what
    distinguishes a ligand that grips the whole series equally from one whose
    cavity is size selective -- exactly the within-extractant signal the
    baseline model is missing.
    """
    r_ion = IONIC_RADIUS.get(metal_symbol, math.nan)
    vecs = geom.coordinates - geom.coordinates[mi]
    dist = np.linalg.norm(vecs, axis=1)
    dist[mi] = np.inf
    is_donor = np.isin(geom.symbols, DONOR_SYMBOLS)
    shell = np.flatnonzero(is_donor & (dist <= DONOR_CUTOFF_A))
    out = {"ionic_radius": float(r_ion)}
    if shell.size == 0 or not np.isfinite(r_ion):
        return out
    sd = np.sort(dist[shell])
    excess = sd - r_ion
    out["excess_mean"] = float(excess.mean())
    out["excess_min"] = float(excess.min())
    out["excess_max"] = float(excess.max())
    out["excess_std"] = float(excess.std())
    out["excess_range"] = float(excess.max() - excess.min())
    # Ratio form -- dimensionless "fit" of the cavity to the cation.
    out["d_over_r_mean"] = float((sd / r_ion).mean())
    out["d_over_r_std"] = float((sd / r_ion).std())
    # Per-donor-element excess: O vs N cavities respond differently to size.
    for element in ("O", "N", "S", "P"):
        mask = geom.symbols[shell] == element
        sel = dist[shell][mask]
        out[f"excess_{element}_mean"] = (
            float((np.sort(sel) - r_ion).mean()) if sel.size else math.nan
        )
    return out


def block_polyhedron(geom: ExtXYZGeometry, mi: int) -> dict[str, float]:
    """G3: shape of the donor polyhedron (scale free)."""
    vecs = geom.coordinates - geom.coordinates[mi]
    dist = np.linalg.norm(vecs, axis=1)
    dist[mi] = np.inf
    is_donor = np.isin(geom.symbols, DONOR_SYMBOLS)
    shell = np.flatnonzero(is_donor & (dist <= DONOR_CUTOFF_A))
    out: dict[str, float] = {}
    if shell.size < 4:
        out.update({f"cshm_{n}": math.nan for n in ALL_SHAPE_NAMES})
        return out
    v = vecs[shell]
    out.update(continuous_shape_measures(v))
    # Angular statistics of the donor set (independent of any reference shape).
    u = v / np.linalg.norm(v, axis=1)[:, None]
    cos = np.clip(u @ u.T, -1.0, 1.0)
    iu = np.triu_indices(len(u), k=1)
    angles = np.degrees(np.arccos(cos[iu]))
    out["angle_mean"] = float(angles.mean())
    out["angle_std"] = float(angles.std())
    out["angle_min"] = float(angles.min())
    out["angle_max"] = float(angles.max())
    # Deviation from a perfectly isotropic donor arrangement: the norm of the
    # sum of unit vectors is 0 for a centrosymmetric shell and 1 for a
    # hemispherically capped one ("open" coordination -> room for water).
    out["shell_anisotropy"] = float(np.linalg.norm(u.sum(axis=0)) / len(u))
    eig, asph, acyl = _inertia_descriptors(u)
    out["shell_eig1"], out["shell_eig2"], out["shell_eig3"] = map(float, eig)
    out["shell_asphericity"] = asph
    out["shell_acylindricity"] = acyl
    # Convex hull of the donors: volume and surface of the coordination cage.
    try:
        hull = ConvexHull(v)
        out["hull_volume"] = float(hull.volume)
        out["hull_area"] = float(hull.area)
        out["hull_sphericity"] = float(
            (np.pi ** (1 / 3)) * (6 * hull.volume) ** (2 / 3) / hull.area
        )
    except Exception:
        out["hull_volume"] = math.nan
        out["hull_area"] = math.nan
        out["hull_sphericity"] = math.nan
    return out


def block_steric(geom: ExtXYZGeometry, mi: int) -> dict[str, float]:
    """G4: how buried / how exposed the metal centre is."""
    coords = geom.coordinates
    radii = np.array([VDW_RADIUS.get(s, DEFAULT_VDW_RADIUS) for s in geom.symbols])
    centre = coords[mi]
    other = np.arange(len(coords)) != mi
    out: dict[str, float] = {}
    for r_sphere in (3.5, 5.0, 7.0):
        out[f"vbur_{str(r_sphere).replace('.', 'p')}"] = buried_volume(
            coords[other], radii[other], centre, r_sphere, n_grid=36
        )
    vecs = coords[other] - centre
    out["solid_angle_frac"] = solid_angle_fraction(vecs, radii[other])
    dist = np.linalg.norm(vecs, axis=1)
    prev = 0.0
    for edge in SHELL_EDGES:
        out[f"n_atoms_within_{str(edge).replace('.', 'p')}"] = float((dist <= edge).sum())
        band = (dist > prev) & (dist <= edge)
        # Local carbon fraction: an all-carbon second shell means the metal is
        # wrapped in a lipophilic jacket -> better organic-phase transfer.
        out[f"cfrac_{str(prev).replace('.', 'p')}_{str(edge).replace('.', 'p')}"] = (
            float(np.mean(geom.symbols[other][band] == "C")) if band.any() else math.nan
        )
        prev = edge
    # Metal SASA -- direct measure of residual accessibility to water.
    areas = shrake_rupley(coords, radii, probe=1.40)
    metal_r = radii[mi]
    out["metal_sasa"] = float(areas[mi])
    out["metal_sasa_frac"] = float(areas[mi] / (4 * np.pi * (metal_r + 1.4) ** 2))
    out["_sasa_cache"] = areas  # consumed by block_global_shape, stripped later
    return out


def block_electronic(geom: ExtXYZGeometry, mi: int, forces: np.ndarray | None
                     ) -> dict[str, float]:
    """G5: xTB charges, charge transfer, dipole geometry, force residuals."""
    out: dict[str, float] = {}
    q = geom.partial_charges
    coords = geom.coordinates
    dist = np.linalg.norm(coords - coords[mi], axis=1)
    dist[mi] = np.inf
    finite_q = np.isfinite(q)
    if finite_q.any():
        out["q_metal"] = float(q[mi]) if np.isfinite(q[mi]) else math.nan
        # Charge transferred from the ligand set into the formal 3+ cation.
        out["q_transfer"] = float(3.0 - q[mi]) if np.isfinite(q[mi]) else math.nan
        is_donor = np.isin(geom.symbols, DONOR_SYMBOLS)
        shell = np.flatnonzero(is_donor & (dist <= DONOR_CUTOFF_A))
        if shell.size:
            qs = q[shell]
            qs = qs[np.isfinite(qs)]
            if qs.size:
                out["q_donor_sum"] = float(qs.sum())
                out["q_donor_mean"] = float(qs.mean())
                out["q_donor_std"] = float(qs.std())
                out["q_donor_min"] = float(qs.min())
                out["q_donor_max"] = float(qs.max())
        for edge in (4.0, 6.0):
            band = dist <= edge
            qb = q[band & finite_q]
            out[f"q_sum_within_{int(edge)}"] = float(qb.sum()) if qb.size else math.nan
        out["q_abs_sum"] = float(np.abs(q[finite_q]).sum())
        out["q_std_all"] = float(q[finite_q].std())
        # Ionic character of the M-L bonds via a point-charge interaction sum.
        is_don = np.isin(geom.symbols, DONOR_SYMBOLS)
        sh = np.flatnonzero(is_don & (dist <= DONOR_CUTOFF_A))
        if sh.size and np.isfinite(q[mi]):
            valid = sh[np.isfinite(q[sh])]
            if valid.size:
                out["coulomb_ML_sum"] = float(
                    (q[mi] * q[valid] / dist[valid]).sum()
                )
    d = geom.dipole
    if np.all(np.isfinite(d)):
        out["dipole_mag"] = float(np.linalg.norm(d))
        # Orientation of the dipole relative to the metal -> ligand-centroid
        # axis: a dipole pointing away from the wrapped side means an exposed,
        # polar face that still needs hydration.
        ligand_centroid = np.delete(coords, mi, axis=0).mean(axis=0)
        axis = ligand_centroid - coords[mi]
        na = np.linalg.norm(axis)
        nd = np.linalg.norm(d)
        if na > 1e-8 and nd > 1e-8:
            out["dipole_cos_axis"] = float(np.dot(d, axis) / (na * nd))
    out["energy_eV"] = float(geom.energy_eV)
    out["free_energy_eV"] = float(geom.free_energy_eV)
    n_atoms = len(geom.symbols)
    out["n_atoms"] = float(n_atoms)
    if np.isfinite(geom.energy_eV):
        out["energy_per_atom"] = float(geom.energy_eV / n_atoms)
    if forces is not None and forces.shape == coords.shape:
        fn = np.linalg.norm(forces, axis=1)
        # Residual forces at the "optimised" geometry = how strained / how
        # unconverged the complex is.  A ligand that cannot relax around a
        # given cation keeps a larger residual.
        out["force_rms"] = float(np.sqrt((fn ** 2).mean()))
        out["force_max"] = float(fn.max())
        out["force_metal"] = float(fn[mi])
        is_donor = np.isin(geom.symbols, DONOR_SYMBOLS)
        sh = np.flatnonzero(is_donor & (dist <= DONOR_CUTOFF_A))
        if sh.size:
            out["force_donor_mean"] = float(fn[sh].mean())
            out["force_donor_max"] = float(fn[sh].max())
    return out


def block_rdf(geom: ExtXYZGeometry, mi: int) -> dict[str, float]:
    """G6: element-resolved metal-centred radial distribution function.

    A smooth, rotation-invariant 3D fingerprint of the environment.  Unlike the
    ordered polyhedron columns it degrades gracefully when the coordination
    number changes, so it transfers between CN 8 and CN 9 complexes.
    """
    coords = geom.coordinates
    dist = np.linalg.norm(coords - coords[mi], axis=1)
    dist[mi] = np.inf
    out: dict[str, float] = {}
    for element in ("C", "N", "O", "H"):
        sel = dist[(geom.symbols == element) & (dist < 10.0)]
        if sel.size == 0:
            for k in range(len(RDF_GRID)):
                out[f"rdf_{element}_{k:02d}"] = 0.0
            continue
        contrib = np.exp(-RDF_ETA * (RDF_GRID[None, :] - sel[:, None]) ** 2)
        out.update({f"rdf_{element}_{k:02d}": float(v)
                    for k, v in enumerate(contrib.sum(axis=0))})
    return out


def block_global_shape(geom: ExtXYZGeometry, mi: int,
                       sasa: np.ndarray | None) -> dict[str, float]:
    """G7: size / shape / surface chemistry of the whole complex.

    log D is a *partition* coefficient, so the exterior of the complex -- how
    big it is, how spherical, and how much of its surface is hydrocarbon versus
    polar -- is at least as relevant as the inner sphere.
    """
    coords = geom.coordinates
    out: dict[str, float] = {}
    rel = coords - coords.mean(axis=0)
    out["rgyr"] = float(np.sqrt((rel ** 2).sum(axis=1).mean()))
    eig, asph, acyl = _inertia_descriptors(coords)
    out["shape_eig1"], out["shape_eig2"], out["shape_eig3"] = map(float, eig)
    out["shape_asphericity"] = asph
    out["shape_acylindricity"] = acyl
    total = eig.sum()
    out["shape_npr1"] = float(eig[0] / eig[2]) if eig[2] > 0 else math.nan
    out["shape_npr2"] = float(eig[1] / eig[2]) if eig[2] > 0 else math.nan
    out["shape_spherocity"] = float(3 * eig[0] / total) if total > 0 else math.nan
    dist_metal = np.linalg.norm(coords - coords[mi], axis=1)
    out["max_radius_from_metal"] = float(dist_metal.max())
    out["metal_offcentre"] = float(np.linalg.norm(coords[mi] - coords.mean(axis=0)))
    try:
        hull = ConvexHull(coords)
        out["complex_hull_volume"] = float(hull.volume)
        out["complex_hull_area"] = float(hull.area)
        out["complex_sphericity"] = float(
            (np.pi ** (1 / 3)) * (6 * hull.volume) ** (2 / 3) / hull.area
        )
        out["packing_density"] = float(len(coords) / hull.volume)
    except Exception:
        pass
    if sasa is not None:
        out["sasa_total"] = float(sasa.sum())
        for element in ("C", "H", "O", "N"):
            mask = geom.symbols == element
            out[f"sasa_{element}"] = float(sasa[mask].sum()) if mask.any() else 0.0
        apolar = np.isin(geom.symbols, ("C", "H"))
        out["sasa_apolar_frac"] = float(sasa[apolar].sum() / sasa.sum()) if sasa.sum() else math.nan
        polar = np.isin(geom.symbols, ("O", "N", "S", "F", "Cl"))
        out["sasa_polar"] = float(sasa[polar].sum())
        # Charge-weighted exposed polarity -- the part of the surface that a
        # water molecule actually sees.
        q = geom.partial_charges
        fin = np.isfinite(q)
        if fin.any():
            out["sasa_weighted_abs_charge"] = float((sasa[fin] * np.abs(q[fin])).sum())
            out["surface_charge_density"] = float(
                (sasa[fin] * np.abs(q[fin])).sum() / sasa.sum()
            ) if sasa.sum() else math.nan
    return out


def _covalent_adjacency(geom: ExtXYZGeometry, scale: float = 1.25) -> np.ndarray:
    r = np.array([COVALENT_RADIUS.get(s, DEFAULT_COVALENT_RADIUS) for s in geom.symbols])
    d = squareform(pdist(geom.coordinates))
    cutoff = scale * (r[:, None] + r[None, :])
    adj = (d < cutoff) & (d > 1e-6)
    return adj


def block_chelate(geom: ExtXYZGeometry, mi: int) -> dict[str, float]:
    """G8: chelate topology -- bite angles and ring sizes of the actual 3D bonds."""
    coords = geom.coordinates
    dist = np.linalg.norm(coords - coords[mi], axis=1)
    dist[mi] = np.inf
    is_donor = np.isin(geom.symbols, DONOR_SYMBOLS)
    shell = np.flatnonzero(is_donor & (dist <= DONOR_CUTOFF_A))
    out: dict[str, float] = {}
    if shell.size < 2:
        return out
    adj = _covalent_adjacency(geom)
    adj[mi, :] = False
    adj[:, mi] = False  # the metal is not a covalent node here
    # BFS distance between donor pairs through the ligand skeleton.
    n = len(coords)
    ring_sizes: list[int] = []
    bite_angles: list[float] = []
    vec = coords[shell] - coords[mi]
    for a in range(len(shell)):
        # BFS from donor a over covalent bonds, capped at depth 6.
        seen = {int(shell[a]): 0}
        frontier = [int(shell[a])]
        for depth in range(1, 7):
            nxt = []
            for node in frontier:
                for nb in np.flatnonzero(adj[node]):
                    nb = int(nb)
                    if nb not in seen:
                        seen[nb] = depth
                        nxt.append(nb)
            frontier = nxt
            if not frontier:
                break
        for b in range(a + 1, len(shell)):
            target = int(shell[b])
            if target in seen:
                ring_sizes.append(seen[target] + 2)  # + metal + closing bond
                v1, v2 = vec[a], vec[b]
                den = np.linalg.norm(v1) * np.linalg.norm(v2)
                if den > 0:
                    bite_angles.append(
                        float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / den, -1, 1))))
                    )
    out["n_chelate_pairs"] = float(len(ring_sizes))
    if ring_sizes:
        rs = np.array(ring_sizes, dtype=float)
        out["ring_size_mean"] = float(rs.mean())
        out["ring_size_min"] = float(rs.min())
        out["ring_size_max"] = float(rs.max())
        out["n_5ring"] = float((rs == 5).sum())
        out["n_6ring"] = float((rs == 6).sum())
        out["n_7ring"] = float((rs == 7).sum())
    if bite_angles:
        ba = np.array(bite_angles)
        out["bite_angle_mean"] = float(ba.mean())
        out["bite_angle_std"] = float(ba.std())
        out["bite_angle_min"] = float(ba.min())
        out["bite_angle_max"] = float(ba.max())
    # Number of distinct ligand fragments bound to the metal (denticity split).
    comp_id = -np.ones(len(coords), dtype=int)
    cid = 0
    for start in range(len(coords)):
        if start == mi or comp_id[start] >= 0:
            continue
        stack = [start]
        comp_id[start] = cid
        while stack:
            node = stack.pop()
            for nb in np.flatnonzero(adj[node]):
                if comp_id[nb] < 0:
                    comp_id[nb] = cid
                    stack.append(int(nb))
        cid += 1
    bound = comp_id[shell]
    out["n_bound_fragments"] = float(len(set(bound.tolist())))
    out["max_denticity"] = float(np.bincount(bound - bound.min()).max()) if bound.size else math.nan
    return out


def _persistence_stats(points: np.ndarray, prefix: str,
                       max_points: int = 400) -> dict[str, float]:
    """G9 helper: H0/H1 persistence summaries of a point cloud."""
    out = {f"{prefix}_{k}": math.nan for k in
           ("h0_total", "h0_max", "h0_entropy", "h0_count",
            "h1_total", "h1_max", "h1_mean", "h1_entropy", "h1_count",
            "h1_birth_mean", "h1_death_mean")}
    try:
        from ripser import ripser
    except Exception:
        return out
    if len(points) < 4:
        return out
    pts = points
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    try:
        dgms = ripser(pts, maxdim=1, thresh=8.0)["dgms"]
    except Exception:
        return out
    for dim, dgm in enumerate(dgms[:2]):
        d = np.asarray(dgm, dtype=float)
        if d.size == 0:
            continue
        d = d[np.isfinite(d).all(axis=1)]
        if d.size == 0:
            continue
        life = d[:, 1] - d[:, 0]
        life = life[life > 0]
        if life.size == 0:
            continue
        p = life / life.sum()
        out[f"{prefix}_h{dim}_total"] = float(life.sum())
        out[f"{prefix}_h{dim}_max"] = float(life.max())
        out[f"{prefix}_h{dim}_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
        out[f"{prefix}_h{dim}_count"] = float(life.size)
        if dim == 1:
            out[f"{prefix}_h1_mean"] = float(life.mean())
            out[f"{prefix}_h1_birth_mean"] = float(d[:, 0].mean())
            out[f"{prefix}_h1_death_mean"] = float(d[:, 1].mean())
    return out


def block_topology(geom: ExtXYZGeometry, mi: int) -> dict[str, float]:
    """G9: persistent-homology summaries (tabular-friendly, unlike PI images)."""
    coords = geom.coordinates
    out: dict[str, float] = {}
    heavy = coords[geom.symbols != "H"]
    out.update(_persistence_stats(heavy, "topo_heavy"))
    dist = np.linalg.norm(coords - coords[mi], axis=1)
    near = coords[(dist <= 6.0)]
    out.update(_persistence_stats(near, "topo_near"))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
BLOCK_PREFIX = {
    "first_shell": "g1",
    "contraction": "g2",
    "polyhedron": "g3",
    "steric": "g4",
    "electronic": "g5",
    "rdf": "g6",
    "global_shape": "g7",
    "chelate": "g8",
    "topology": "g9",
}


def _read_forces(path: Path, n_atoms: int) -> np.ndarray | None:
    """Pull the ``forces:R:3`` columns out of the extxyz property layout."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        header = lines[1]
        spec = None
        for token in header.split():
            if token.startswith("Properties="):
                spec = token.split("=", 1)[1]
                break
        if spec is None:
            return None
        parts = spec.split(":")
        col = 0
        start = None
        for i in range(0, len(parts) - 2, 3):
            name, _kind, width = parts[i], parts[i + 1], int(parts[i + 2])
            if name == "forces":
                start = col
                break
            col += width
        if start is None:
            return None
        rows = []
        for line in lines[2:2 + n_atoms]:
            v = line.split()
            rows.append([float(v[start]), float(v[start + 1]), float(v[start + 2])])
        return np.asarray(rows, dtype=float)
    except Exception:
        return None


def features_for_geometry(path: Path, metal_symbol: str | None,
                          blocks: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Compute every enabled descriptor block for a single geometry file."""
    wanted = set(blocks) if blocks else set(BLOCK_PREFIX)
    geom = read_extxyz(path)
    mi = _metal_index(geom, metal_symbol)
    forces = _read_forces(path, len(geom.symbols))
    row: dict[str, Any] = {}
    sasa_cache = None

    def stash(name: str, values: dict[str, float]) -> None:
        pref = BLOCK_PREFIX[name]
        for k, v in values.items():
            if k.startswith("_"):
                continue
            row[f"{pref}__{name}__{k}"] = v

    if "first_shell" in wanted:
        stash("first_shell", block_first_shell(geom, mi))
    if "contraction" in wanted:
        stash("contraction", block_contraction(geom, mi, geom.symbols[mi]))
    if "polyhedron" in wanted:
        stash("polyhedron", block_polyhedron(geom, mi))
    if "steric" in wanted:
        st = block_steric(geom, mi)
        sasa_cache = st.pop("_sasa_cache", None)
        stash("steric", st)
    if "electronic" in wanted:
        stash("electronic", block_electronic(geom, mi, forces))
    if "rdf" in wanted:
        stash("rdf", block_rdf(geom, mi))
    if "global_shape" in wanted:
        if sasa_cache is None:
            radii = np.array([VDW_RADIUS.get(s, DEFAULT_VDW_RADIUS) for s in geom.symbols])
            sasa_cache = shrake_rupley(geom.coordinates, radii, probe=1.40)
        stash("global_shape", block_global_shape(geom, mi, sasa_cache))
    if "chelate" in wanted:
        stash("chelate", block_chelate(geom, mi))
    if "topology" in wanted:
        stash("topology", block_topology(geom, mi))
    return row


def build_geometry_table(jobs: pd.DataFrame, blocks: tuple[str, ...] | None = None,
                         verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``features_for_geometry`` over a job table of (geometry_key, path)."""
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    t0 = time.time()
    for i, rec in enumerate(jobs.itertuples(index=False)):
        try:
            feats = features_for_geometry(Path(rec.local_xyz_path), rec.metal, blocks)
            feats["geometry_key"] = rec.geometry_key
            rows.append(feats)
        except Exception as exc:  # keep the sweep alive, record the failure
            failures.append({
                "geometry_key": rec.geometry_key,
                "local_xyz_path": str(rec.local_xyz_path),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
            })
        if verbose and (i + 1) % 25 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"[geom3d] {i + 1}/{len(jobs)} ok={len(rows)} fail={len(failures)} "
                  f"{rate:.2f} geom/s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(failures)


def resolve_jobs(repo_root: Path, only_ok: bool = False,
                 geom_root: Path | None = None) -> pd.DataFrame:
    """Map every dataset geometry_key to a local xyz file (matched by basename).

    The parquet stores absolute paths from the *original* cluster checkout, so
    resolution is by file name against the local geometry tree.

    ``geom_root`` selects **which** geometry set to featurise.  This matters:
    the re-optimised structures under ``automl/artifacts/geom_reopt/<solvent>/``
    deliberately reuse the original basenames, and this resolver is
    first-match-wins over an unordered ``rglob``.  Without an explicit root, a
    run could silently featurise a mixture of loose and re-optimised geometries
    and the resulting descriptors would correspond to no single geometry set at
    all.  The default therefore *excludes* ``geom_reopt`` entirely, so the
    original behaviour is exactly preserved and the new geometries are only
    ever used when asked for by name.
    """
    disk: dict[str, Path] = {}
    if geom_root is not None:
        for p in sorted(Path(geom_root).rglob("*.xyz")):
            disk.setdefault(p.name, p)
    else:
        for p in sorted(repo_root.rglob("*.xyz")):
            if "geom_reopt" in p.parts:
                continue
            disk.setdefault(p.name, p)
    df = pd.read_parquet(repo_root / "data/processed/final_ml_dataset_3d.parquet",
                         columns=["geometry_key", "xyz_path", "metal", "geometry_ok",
                                  "geometry_qc_class"])
    if only_ok:
        df = df[df["geometry_ok"]]
    df = df.dropna(subset=["xyz_path"]).drop_duplicates("geometry_key")
    df["basename"] = df["xyz_path"].map(lambda p: os.path.basename(str(p)))
    df["local_xyz_path"] = df["basename"].map(lambda b: str(disk[b]) if b in disk else None)
    return df.dropna(subset=["local_xyz_path"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--blocks", type=str, default="",
                    help="comma separated subset of " + ",".join(BLOCK_PREFIX))
    ap.add_argument("--only-ok", action="store_true",
                    help="restrict to qc_class == OK geometries")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--geom-root", type=str, default="",
                    help="featurise geometries from this directory instead of "
                         "the default tree (e.g. a geom_reopt/<solvent> dir)")
    args = ap.parse_args()

    blocks = tuple(b for b in args.blocks.split(",") if b) or None
    jobs = resolve_jobs(_REPO_ROOT, only_ok=args.only_ok,
                        geom_root=Path(args.geom_root) if args.geom_root else None)
    if args.limit:
        jobs = jobs.head(args.limit)
    jobs = jobs.iloc[args.shard::args.num_shards].reset_index(drop=True)
    print(f"[geom3d] shard {args.shard}/{args.num_shards}: {len(jobs)} geometries",
          flush=True)

    table, failures = build_geometry_table(jobs, blocks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    table.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    if len(failures):
        failures.to_csv(out.with_name(out.stem + "_failures.csv"), index=False)
    print(json.dumps({
        "shard": args.shard,
        "n_jobs": len(jobs),
        "n_ok": int(len(table)),
        "n_fail": int(len(failures)),
        "n_features": int(table.shape[1] - 1) if len(table) else 0,
        "out": str(out),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
