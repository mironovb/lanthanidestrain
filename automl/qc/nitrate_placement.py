#!/usr/bin/env python3
"""Place NO3- counter-ions around an existing complex without touching it.

Why this is a separate module
-----------------------------
The placement arithmetic is the part that can produce a *plausible-looking wrong
structure*, so it is kept free of xTB and of file I/O and is unit-tested on its
own.  Everything here is deterministic: no RNG, no wall-clock, no dict ordering.

Design decisions that are NOT arbitrary, with the measurement behind each:

* **Outer-sphere only.**  100 % of the 956 complexes sit at coordination number
  >= 8 at this repo's own 3.10 A donor cutoff (63 % at CN 9): the ligand already
  saturates the sphere.  A rigid inner-sphere insertion search over the
  nitrate-free structures found a best-pose clearance of 1.55 A against the 2.75 A
  that real bound nitrates in this corpus actually achieve -- feasible fraction
  0.000.  Seeding inner-sphere anyway yields a converged, correct-formula
  structure whose ligand has been reorganised, which is the worst possible
  artefact because it looks fine.

* **Surface-referenced radius, never a fixed one.**  Median molecular extent from
  the metal is 10.7 A and 98 % of complexes have atoms beyond 6 A, so a nitrate
  seeded at a nominal "5-6 A outer sphere" lands inside the ligand backbone.

* **Clash floors are empirical, not van der Waals sums.**  vdW-sum clearance
  scores real, relaxed, published nitrate positions at about -1.5 A, because
  coordination bonds violate vdW radii by construction.  A vdW criterion rejects
  correct chemistry.

* **Free-ion D3h template.**  All three N-O equal.  The 1.231/1.198 asymmetry
  measured in the corpus is *caused* by chelation and would be a wrong seed for a
  second-sphere ion.
"""

from __future__ import annotations

import numpy as np

# --- NO3- free-ion template ------------------------------------------------
# Symmetric mean of the corpus's chelating (1.231) and distal (1.198) N-O bonds.
NO3_NO_BOND_A = 1.224
NO3_ONO_DEG = 120.0

# --- clash floors, from what real bound nitrates in this corpus contact -----
OUTER_MIN_HEAVY_A = 3.10      # ordinary non-bonded O...C/O...O contact
OUTER_MIN_H_A = 2.30          # permits a C-H...O second-sphere H-bond
R_GRID_MIN_A, R_GRID_MAX_A, R_GRID_STEP_A = 4.0, 16.0, 0.25

N_FIB = 512                   # ~5.1 deg mean nearest-neighbour spacing
N_ROT = 12                    # pi/12 steps; the fragment has a C2 axis
PLANE_MODES = ("edge", "face")
MIN_NLN_SEPARATION_DEG = 90.0  # two -1 ions must not be seeded into contact

PRUNE_RADIUS_A = 8.0          # obstacles beyond r_place + this are ignored


def fibonacci_sphere(n: int = N_FIB) -> np.ndarray:
    """Deterministic near-uniform directions. No RNG, so nothing to reproduce."""
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    theta = np.pi * (1.0 + np.sqrt(5.0)) * (i + 0.5)
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


def orthonormal_frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame perpendicular to u.

    The |u_z| > 0.9 branch is essential: a naive cross(u, z_hat) degenerates at
    the poles and silently yields a NaN frame, which would propagate into
    coordinates rather than raising.
    """
    a = np.array([1.0, 0.0, 0.0]) if abs(u[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    e2 = np.cross(u, a)
    n = np.linalg.norm(e2)
    if n < 1e-12:                       # unreachable given the branch, asserted anyway
        raise ValueError("degenerate frame")
    e2 = e2 / n
    return e2, np.cross(u, e2)


def nitrate_template() -> tuple[list[str], np.ndarray]:
    """Free D3h nitrate with N at the origin, ion plane = local xy."""
    ang = np.deg2rad(np.array([90.0, 210.0, 330.0]))
    o = np.column_stack([NO3_NO_BOND_A * np.cos(ang),
                         NO3_NO_BOND_A * np.sin(ang),
                         np.zeros(3)])
    return ["N", "O", "O", "O"], np.vstack([np.zeros(3), o])


def _oriented(u, e2, e3, t: float, plane: str, centre: np.ndarray) -> np.ndarray:
    """Template rotated by t about u, in the requested plane mode, at centre."""
    _, tpl = nitrate_template()
    c, s = np.cos(t), np.sin(t)
    if plane == "face":                  # ion plane perpendicular to u
        b1, b2, b3 = e2, e3, u
    else:                                # edge-on: ion plane contains u
        b1, b2, b3 = u, e2, e3
    r1 = c * b1 + s * b2
    r2 = -s * b1 + c * b2
    basis = np.vstack([r1, r2, b3])      # rows map local xyz -> lab
    return centre + tpl @ basis


NO3_RADIUS_A = 1.30           # N-O bond + margin: bounds the fragment about N


def _min_dists(points: np.ndarray, obst: np.ndarray, is_h: np.ndarray):
    """(min heavy dist, min H dist) per point. One vectorised pass."""
    d = np.linalg.norm(points[:, None, :] - obst[None, :, :], axis=2)
    dh = d[:, ~is_h].min(axis=1) if (~is_h).any() else np.full(len(points), np.inf)
    dy = d[:, is_h].min(axis=1) if is_h.any() else np.full(len(points), np.inf)
    return dh, dy


def place_one(coords: np.ndarray, symbols: list[str], metal_idx: int,
              extra: np.ndarray | None = None,
              placed_dirs: list[np.ndarray] | None = None) -> dict:
    """Best outer-sphere pose for one nitrate.

    Two stages, because the exact search is 512 x 12 x 2 x 48 poses and that is
    far too slow in Python:

    1. Screen CENTRES only, vectorised.  A centre whose clearance is below
       ``floor + NO3_RADIUS_A`` cannot host the fragment in any orientation, so
       the whole ray is eliminated without touching an orientation.
    2. Evaluate the surviving (direction, radius) pairs exactly, over all
       orientations, again vectorised.

    Stage 1 is a strict lower bound, so it can only discard poses that stage 2
    would also have rejected -- it changes the cost, never the answer.

    Returns the pose AND the numbers that explain it.  ``n_feasible`` in
    particular distinguishes "no room at all" from "room, but the relaxation
    ruined it" when a structure is later rejected.
    """
    metal = np.asarray(coords[metal_idx], dtype=float)
    parts = [np.asarray(coords, dtype=float)]
    sym = list(symbols)
    if extra is not None and len(extra):
        parts.append(np.asarray(extra, dtype=float))
        sym = sym + ["O"] * len(extra)          # placed nitrates: heavy obstacles
    obst = np.vstack(parts)
    keep = np.ones(len(obst), dtype=bool)
    keep[metal_idx] = False                     # the metal is the reference
    obst = obst[keep]
    is_h = np.array([a == "H" for a in sym])[keep]
    near = np.linalg.norm(obst - metal, axis=1) <= (R_GRID_MAX_A + PRUNE_RADIUS_A)
    obst, is_h = obst[near], is_h[near]

    dirs = fibonacci_sphere()
    if placed_dirs:
        cosmin = np.cos(np.deg2rad(MIN_NLN_SEPARATION_DEG))
        okdir = np.array([not any(float(np.dot(u, v)) > cosmin for v in placed_dirs)
                          for u in dirs])
        idx_keep = np.flatnonzero(okdir)
    else:
        idx_keep = np.arange(len(dirs))
    if idx_keep.size == 0:
        return dict(ok=False, reason="SEED_NO_FEASIBLE_POSE", n_feasible=0)

    radii = np.arange(R_GRID_MIN_A, R_GRID_MAX_A + 1e-9, R_GRID_STEP_A)
    # ---- stage 1: centres only -------------------------------------------
    centres = (metal[None, None, :]
               + radii[None, :, None] * dirs[idx_keep][:, None, :])
    flat = centres.reshape(-1, 3)
    dh, dy = _min_dists(flat, obst, is_h)
    lb = np.minimum(dh - OUTER_MIN_HEAVY_A, dy - OUTER_MIN_H_A) - NO3_RADIUS_A
    lb = lb.reshape(len(idx_keep), len(radii))
    cand = []
    for a, row in enumerate(lb):
        f = np.flatnonzero(row >= 0)
        if f.size:
            cand.append((int(idx_keep[a]), int(f[0])))   # snuggest feasible r
    if not cand:
        return dict(ok=False, reason="SEED_NO_FEASIBLE_POSE", n_feasible=0)

    # ---- stage 2: exact, over orientations -------------------------------
    best = None
    n_feasible = 0
    for di, ri in cand:
        u = dirs[di]
        e2, e3 = orthonormal_frame(u)
        centre = metal + radii[ri] * u
        poses, meta = [], []
        for k in range(N_ROT):
            t = np.pi * k / N_ROT
            for plane in PLANE_MODES:
                poses.append(_oriented(u, e2, e3, t, plane, centre))
                meta.append((k, plane))
        P = np.vstack(poses)
        dh2, dy2 = _min_dists(P, obst, is_h)
        s_all = np.minimum(dh2 - OUTER_MIN_HEAVY_A,
                           dy2 - OUTER_MIN_H_A).reshape(len(meta), 4).min(axis=1)
        dh_m = dh2.reshape(len(meta), 4).min(axis=1)
        dy_m = dy2.reshape(len(meta), 4).min(axis=1)
        good = np.flatnonzero(s_all >= 0)
        n_feasible += int(good.size)
        for g in good:
            k, plane = meta[g]
            # SNUGGEST first, not most open.  Maximising clearance selects the
            # direction with the most empty space, i.e. it pushes the ion away
            # from the complex entirely -- measured: median seed radius 10.5 A
            # at a clearance of 1.54 A, which is a detached ion in vacuum, not a
            # counter-ion associated with this complex.  The floor already
            # guarantees no clash, so the objective is closest approach, with
            # clearance only as a tiebreak.
            key = (round(float(radii[ri]), 6), -round(float(s_all[g]), 6),
                   di, k, plane)
            if best is None or key < best[0]:
                best = (key, dict(pos=poses[g], u=u, fib_index=di, rot_index=k,
                                  plane_mode=plane, r_seed=float(radii[ri]),
                                  clearance=float(s_all[g]),
                                  min_heavy=float(dh_m[g]), min_h=float(dy_m[g])))
    if best is None:
        return dict(ok=False, reason="SEED_NO_FEASIBLE_POSE", n_feasible=0)
    rec = best[1]
    rec.update(ok=True, n_feasible=n_feasible)
    return rec
