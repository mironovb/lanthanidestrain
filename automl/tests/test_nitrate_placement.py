#!/usr/bin/env python3
"""The placement arithmetic, tested without xTB and without file I/O.

This is the code that can produce a *plausible-looking wrong structure*: a
converged, correct-formula complex whose ligand has been reorganised, or a
counter-ion floating in vacuum 14 A away.  Neither is visible in a composition
check, so the properties are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from automl.qc.nitrate_placement import (NO3_NO_BOND_A, NO3_ONO_DEG,
                                         OUTER_MIN_HEAVY_A, OUTER_MIN_H_A,
                                         fibonacci_sphere, nitrate_template,
                                         orthonormal_frame, place_one)


def test_nitrate_template_reproduces_its_own_constants():
    """A mistyped constant is caught here, not three CPU-days later."""
    sym, xyz = nitrate_template()
    assert sym == ["N", "O", "O", "O"]
    for i in (1, 2, 3):
        assert np.linalg.norm(xyz[i] - xyz[0]) == pytest.approx(NO3_NO_BOND_A, abs=1e-9)
    angs = []
    for a, b in ((1, 2), (2, 3), (3, 1)):
        v1, v2 = xyz[a] - xyz[0], xyz[b] - xyz[0]
        angs.append(np.degrees(np.arccos(
            np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))))
    for a in angs:
        assert a == pytest.approx(NO3_ONO_DEG, abs=1e-9)
    assert sum(angs) == pytest.approx(360.0, abs=1e-9)
    # exactly planar: a pyramidalised seed would relax somewhere else entirely
    assert np.abs(xyz[:, 2]).max() == 0.0


def test_free_ion_template_is_symmetric():
    """A second-sphere ion is not chelating.

    The corpus's 1.231/1.198 N-O asymmetry is CAUSED by chelation, so seeding it
    on a free ion would be a wrong starting structure.
    """
    _, xyz = nitrate_template()
    d = [np.linalg.norm(xyz[i] - xyz[0]) for i in (1, 2, 3)]
    assert max(d) - min(d) == pytest.approx(0.0, abs=1e-12)


def test_orthonormal_frame_survives_the_poles():
    """cross(u, z_hat) degenerates at the poles and yields a NaN frame silently."""
    for u in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
              np.array([0.0, 0.0, 0.95]) / np.linalg.norm([0, 0, 0.95]),
              np.array([1.0, 0.0, 0.0])):
        e2, e3 = orthonormal_frame(u)
        assert np.isfinite(e2).all() and np.isfinite(e3).all()
        assert abs(float(np.dot(u, e2))) < 1e-12
        assert abs(float(np.dot(u, e3))) < 1e-12
        assert np.linalg.norm(e2) == pytest.approx(1.0, abs=1e-12)


def test_fibonacci_sphere_is_deterministic_and_unit():
    a, b = fibonacci_sphere(), fibonacci_sphere()
    np.testing.assert_array_equal(a, b)      # no RNG anywhere
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12)


def _toy():
    """A small rigid 'complex': metal at origin, a shell of carbons at 2.4 A."""
    dirs = fibonacci_sphere(24)
    coords = np.vstack([np.zeros(3), dirs * 2.4])
    symbols = ["La"] + ["C"] * 24
    return coords, symbols, 0


def test_placement_never_clashes():
    """The returned pose must satisfy the floors it was selected under."""
    coords, symbols, mi = _toy()
    rec = place_one(coords, symbols, mi)
    assert rec["ok"], rec
    obst = coords[1:]
    d = np.linalg.norm(rec["pos"][:, None, :] - obst[None, :, :], axis=2)
    assert d.min() >= OUTER_MIN_HEAVY_A - 1e-9


def test_placement_is_bit_reproducible():
    """Symmetric complexes have degenerate optima; the tiebreak must be exact.

    Without the deterministic index tiebreak, float noise across BLAS builds
    would silently produce a different artefact from the same input.
    """
    coords, symbols, mi = _toy()
    a = place_one(coords, symbols, mi)
    b = place_one(coords, symbols, mi)
    np.testing.assert_array_equal(a["pos"], b["pos"])
    assert (a["fib_index"], a["rot_index"], a["plane_mode"]) == \
           (b["fib_index"], b["rot_index"], b["plane_mode"])


def test_placement_prefers_the_snuggest_pocket_not_open_space():
    """Maximising clearance puts the ion in vacuum; measured at 10.5 A.

    The floor already guarantees no clash, so the objective is closest approach.
    This pins that the chosen radius is near the smallest feasible one.
    """
    coords, symbols, mi = _toy()
    rec = place_one(coords, symbols, mi)
    assert rec["ok"]
    # the shell is at 2.4 A; a snug pose sits just outside it, not far away
    assert rec["r_seed"] < 8.0, f"ion placed at {rec['r_seed']} A -- detached"


def test_second_nitrate_is_kept_away_from_the_first():
    """Two -1 ions must not be seeded into a contact pair."""
    coords, symbols, mi = _toy()
    first = place_one(coords, symbols, mi)
    assert first["ok"]
    second = place_one(coords, symbols, mi, extra=first["pos"],
                       placed_dirs=[first["u"]])
    assert second["ok"], second
    cosang = float(np.dot(first["u"], second["u"]))
    assert cosang <= np.cos(np.deg2rad(90.0)) + 1e-9
    # and it must not clash with the already-placed fragment
    d = np.linalg.norm(second["pos"][:, None, :] - first["pos"][None, :, :], axis=2)
    assert d.min() >= OUTER_MIN_HEAVY_A - 1e-9


def test_no_feasible_pose_is_reported_not_guessed():
    """A fully enclosed metal must return a reason, never a clashing pose."""
    dirs = fibonacci_sphere(200)
    coords = np.vstack([np.zeros(3)] + [dirs * r for r in np.arange(2.0, 18.0, 0.6)])
    symbols = ["La"] + ["C"] * (len(coords) - 1)
    rec = place_one(coords, symbols, 0)
    assert rec["ok"] is False
    assert rec["reason"] == "SEED_NO_FEASIBLE_POSE"
    assert rec["n_feasible"] == 0
