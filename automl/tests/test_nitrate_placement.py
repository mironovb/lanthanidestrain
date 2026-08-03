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


# ---------------------------------------------------------------------------
# Gate tests.  Synthetic structures written to a tmp dir -- never generated
# artefacts, per AGENTS.md.

def _write_xyz(path, symbols, coords):
    lines = [str(len(symbols)), "test"]
    for s, c in zip(symbols, coords):
        lines.append(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")


def _synthetic(tmp_path, *, h_offset=None, break_no=False, pyramidal=False,
               inner=False):
    """A minimal La complex plus one outer-sphere nitrate, perturbable."""
    from automl.qc.nitrate_placement import nitrate_template
    # La at origin, 8 O donors at 2.4 A, 1 H far away
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                     [0, 0, 1], [0, 0, -1], [0.7, 0.7, 0], [-0.7, -0.7, 0]],
                    dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sym = ["La"] + ["O"] * 8 + ["H"]
    xyz = np.vstack([np.zeros(3), dirs * 2.4, np.array([0.0, 0.0, 5.0])])
    nsym, ntpl = nitrate_template()
    centre = np.array([7.0, 0.0, 0.0]) if not inner else np.array([2.5, 0.0, 0.0])
    nxyz = ntpl + centre
    if break_no:
        nxyz[1] = nxyz[0] + (nxyz[1] - nxyz[0]) * 1.6      # stretched N-O
    if pyramidal:
        nxyz[0] = nxyz[0] + np.array([0.0, 0.0, 0.45])     # N out of the O3 plane
    if h_offset is not None:
        xyz[9] = nxyz[1] + np.array([h_offset, 0.0, 0.0])  # H onto a nitrate O
    return sym + nsym, np.vstack([xyz, nxyz])


def _record(tmp_path, sym, xyz, n0, control_xyz=None):
    """control_xyz lets a test perturb the NEUTRAL arm only.

    Without it both arms are written from the same coordinates, so a
    connectivity change appears in the control too and cancels -- which is how
    this helper silently made the G4c test vacuous.
    """
    n = tmp_path / "neutral.xyz"; c = tmp_path / "control.xyz"
    _write_xyz(n, sym, xyz)
    _write_xyz(c, sym[:n0], (xyz if control_xyz is None else control_xyz)[:n0])
    import automl.qc.neutralize_report as R
    R.REPO = tmp_path
    return {"geometry_key": "t", "metal": "La", "n_add": 1,
            "cn_ligand_in": 8, "neutral_xyz": "neutral.xyz",
            "control_xyz": "control.xyz",
            "neutral": {"xtb_converged": True, "meets_target": True},
            "control": {"xtb_converged": True, "meets_target": True}}


def test_gates_accept_a_clean_synthetic_structure(tmp_path):
    from automl.qc.neutralize_report import gate_structure
    sym, xyz = _synthetic(tmp_path)
    g = gate_structure(_record(tmp_path, sym, xyz, 10))
    assert g["reject_code"] == "accepted", g


def test_g2_rejects_proton_transfer(tmp_path):
    """The most dangerous silent failure in the pipeline.

    A -1 anion against an acidic proton gives HNO3 + deprotonated ligand: same
    formula, same total charge, converged, plausible -- and invisible to every
    composition or charge check. Only the H...O distance sees it.
    """
    from automl.qc.neutralize_report import gate_structure
    sym, xyz = _synthetic(tmp_path, h_offset=0.98)   # H bonded to a nitrate O
    g = gate_structure(_record(tmp_path, sym, xyz, 10))
    assert g["reject_code"] == "NITRATE_PROTONATED", g
    assert g["min_H_to_nitrate_O"] < 1.40


def test_g2_rejects_a_broken_nitrate(tmp_path):
    from automl.qc.neutralize_report import gate_structure
    sym, xyz = _synthetic(tmp_path, break_no=True)
    g = gate_structure(_record(tmp_path, sym, xyz, 10))
    assert g["reject_code"] == "NITRATE_BROKEN", g


def test_g2_rejects_a_pyramidalised_nitrate(tmp_path):
    from automl.qc.neutralize_report import gate_structure
    sym, xyz = _synthetic(tmp_path, pyramidal=True)
    g = gate_structure(_record(tmp_path, sym, xyz, 10))
    assert g["reject_code"] == "NITRATE_PYRAMIDAL", g


def test_inner_sphere_migration_is_accepted_and_recorded(tmp_path):
    """Amendment 1: this is the correct outcome, not a failure.

    The pilot showed nitrates seeded at ~6 A relaxing to Ln-O of 2.13-2.53 A
    with cn_ligand unchanged -- textbook bidentate coordination.  Rejecting it
    would have discarded exactly the physics the campaign exists to capture, so
    the mode is recorded as a covariate instead of used as a filter.
    """
    from automl.qc.neutralize_report import gate_structure
    sym, xyz = _synthetic(tmp_path, inner=True)
    g = gate_structure(_record(tmp_path, sym, xyz, 10))
    assert g["reject_code"] == "accepted", g
    assert g["binding_modes"] == "inner"
    assert g["n_inner_nitrate"] == 1


def test_a_detached_ion_is_still_rejected(tmp_path):
    """Accepting inner-sphere must not also accept an ion in vacuum."""
    from automl.qc.neutralize_report import gate_structure
    from automl.qc.nitrate_placement import nitrate_template
    sym, xyz = _synthetic(tmp_path)
    _, tpl = nitrate_template()
    xyz[10:14] = tpl + np.array([30.0, 0.0, 0.0])      # 30 A away
    g = gate_structure(_record(tmp_path, sym, xyz, 10))
    assert g["reject_code"] == "ION_DETACHED", g


def test_g4c_rejects_a_broken_ligand_bond(tmp_path):
    """The gate accommodation cannot explain away.

    RMSD and donor shifts legitimately move when the nitrate coordinates, so
    connectivity is what remains load-bearing: no bond may break or form
    anywhere in the ligand.
    """
    from automl.qc.neutralize_report import gate_structure
    sym, clean = _synthetic(tmp_path)
    sym2, xyz2 = _synthetic(tmp_path)
    # move the ligand H onto a donor O in the NEUTRAL arm only: forms a bond the
    # control does not have
    xyz2[9] = xyz2[1] + np.array([0.0, 0.0, 0.98])
    g = gate_structure(_record(tmp_path, sym2, xyz2, 10, control_xyz=clean))
    assert g["reject_code"] == "CONNECTIVITY_CHANGED", g
    assert g["adjacency_changed_pairs"] >= 1
