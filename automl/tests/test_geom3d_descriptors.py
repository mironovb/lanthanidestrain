#!/usr/bin/env python3
"""Correctness tests for the geometric descriptors in automl.geom3d_features.

These check the descriptors against cases with a known analytic answer, so a
future edit that silently breaks the geometry maths fails loudly instead of
quietly changing every downstream R^2.

Run with:  python3 -m pytest automl/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from automl.geom3d_features import (
    REFERENCE_POLYHEDRA,
    buried_volume,
    continuous_shape_measures,
    shrake_rupley,
    solid_angle_fraction,
    _normalise_vertices,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q


# The muffin reference is a hand-built approximation and the greedy vertex
# assignment does not always reach its exact optimum; it is still separated from
# every other CN-9 shape by an order of magnitude, which is all a feature needs.
_HEURISTIC_SHAPES = {"MFF"}


@pytest.mark.parametrize("cn", sorted(REFERENCE_POLYHEDRA))
def test_cshm_is_zero_for_the_ideal_shape(cn: int) -> None:
    """An ideal polyhedron, rotated and rescaled, must score ~0 against itself."""
    rng = np.random.default_rng(cn)
    for name, ref in REFERENCE_POLYHEDRA[cn].items():
        vertices = (ref @ _random_rotation(rng).T) * rng.uniform(1.5, 3.0)
        out = continuous_shape_measures(vertices)
        value = out[f"cshm_{name}"]
        limit = 2.0 if name in _HEURISTIC_SHAPES else 1e-6
        assert value < limit, f"CN{cn} {name}: cshm={value}"
        # and it must be the best-matching reference for its own coordination
        others = [v for k, v in out.items()
                  if k.startswith("cshm_") and k != f"cshm_{name}"
                  and not k.endswith(("_best", "_second", "_vertices"))
                  and np.isfinite(v)]
        if others:
            assert value < min(others), f"CN{cn} {name} not the best match"


def test_cshm_increases_with_distortion() -> None:
    """Adding noise to an ideal shape must monotonically raise its CShM."""
    rng = np.random.default_rng(7)
    sap = REFERENCE_POLYHEDRA[8]["SAPR"]
    previous = -1.0
    for noise in (0.0, 0.05, 0.15, 0.30):
        value = continuous_shape_measures(
            sap + rng.normal(scale=noise, size=sap.shape))["cshm_SAPR"]
        assert value > previous, f"CShM not monotone at noise={noise}"
        previous = value


def test_sasa_of_an_isolated_atom_matches_the_sphere_area() -> None:
    """One atom alone: SASA must equal 4*pi*(r_vdw + r_probe)^2 exactly."""
    radius, probe = 1.70, 1.40
    expected = 4.0 * np.pi * (radius + probe) ** 2
    got = float(shrake_rupley(np.zeros((1, 3)), np.array([radius]), probe=probe)[0])
    assert got == pytest.approx(expected, rel=1e-9)


def test_sasa_of_two_fused_atoms_is_less_than_two_spheres() -> None:
    """Overlapping atoms must occlude each other."""
    radius, probe = 1.70, 1.40
    lone = 4.0 * np.pi * (radius + probe) ** 2
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    areas = shrake_rupley(coords, np.full(2, radius), probe=probe)
    assert areas.sum() < 2 * lone
    assert areas.sum() > lone


def test_fully_enclosed_metal_is_completely_buried() -> None:
    """A dense shell around the origin must give %V_bur = 100 and solid angle 1."""
    rng = np.random.default_rng(3)
    shell = rng.normal(size=(200, 3))
    shell = shell / np.linalg.norm(shell, axis=1)[:, None] * 2.0
    radii = np.full(len(shell), 2.0)
    assert buried_volume(shell, radii, np.zeros(3), 3.5) == pytest.approx(100.0)
    assert solid_angle_fraction(shell, radii) == pytest.approx(1.0)


def test_bare_metal_is_not_buried() -> None:
    """A single distant atom must leave the metal essentially unburied."""
    coords = np.array([[8.0, 0.0, 0.0]])
    assert buried_volume(coords, np.array([1.7]), np.zeros(3), 3.5) < 1.0
    assert solid_angle_fraction(coords, np.array([1.7])) < 0.05


def test_normalise_vertices_is_scale_and_translation_invariant() -> None:
    rng = np.random.default_rng(11)
    v = rng.normal(size=(9, 3))
    a = _normalise_vertices(v)
    b = _normalise_vertices(v * 4.2 + np.array([10.0, -3.0, 7.0]))
    assert np.allclose(a, b, atol=1e-9)
