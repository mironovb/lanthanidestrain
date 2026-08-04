#!/usr/bin/env python3
"""The three campaign-4 geometry assets must be interchangeable, or nothing compares.

Skips cleanly when the assets are absent, so a fresh checkout still passes.

The two invariants here are the ones that fail SILENTLY.  A mismatched build_id
list makes the paired bootstrap compare two different row sets; a shared
triangle-edge cache makes an asset load the wrong boundary map.  Neither raises,
and both produce a run that looks entirely normal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "automl/artifacts/vr_neutral"
ARMS = ("shipped", "control", "neutral")


def _asset(arm: str) -> Path:
    return ROOT / arm / "vietoris_rips_inputs.npz"


def _available() -> list[str]:
    return [a for a in ARMS if _asset(a).exists()]


def test_all_arms_carry_identical_build_ids():
    """Or the paired bootstrap compares two different populations."""
    have = _available()
    if len(have) < 2:
        pytest.skip("fewer than two geometry assets built")
    ref = None
    for arm in have:
        ids = [str(b) for b in np.load(_asset(arm))["build_ids"]]
        if ref is None:
            ref = (arm, ids)
            continue
        assert ids == ref[1], (
            f"{arm} and {ref[0]} carry different build_ids "
            f"({len(ids)} vs {len(ref[1])}); every cross-arm contrast would "
            f"compare two different row sets")


def test_each_arm_has_its_own_triangle_cache_path():
    """SimplicialComplexes keys that cache only by triangle count.

    An asset loaded without its own path silently reuses the shipped boundary
    map, and every downstream number is quietly wrong.
    """
    from automl.topo.train import geometry_asset
    have = _available()
    if not have:
        pytest.skip("no geometry assets built")
    for arm in have:
        cache = ROOT / arm / "triangle_edges.npz"
        S = geometry_asset(arm)
        assert len(S) > 0
        # the loader must have pointed at this arm's own cache, not the shipped one
        assert cache.exists(), (
            f"{arm} produced no triangle cache at {cache}; it is reusing "
            f"another asset's boundary map")


def test_shipped_arm_matches_the_shipped_asset_where_they_overlap():
    """The shipped arm is a SUBSET copy, not a re-derivation."""
    if not _asset("shipped").exists():
        pytest.skip("shipped arm not built")
    ship = np.load(REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz")
    arm = np.load(_asset("shipped"))
    idx = {str(b): i for i, b in enumerate(ship["build_ids"])}
    ids = [str(b) for b in arm["build_ids"]]
    checked = 0
    for j, bid in enumerate(ids[:25]):
        assert bid in idx, f"{bid} is not in the shipped asset"
        i = idx[bid]
        a = slice(int(arm["node_ptr"][j]), int(arm["node_ptr"][j + 1]))
        b = slice(int(ship["node_ptr"][i]), int(ship["node_ptr"][i + 1]))
        np.testing.assert_array_equal(arm["atomic_numbers"][a],
                                      ship["atomic_numbers"][b])
        np.testing.assert_allclose(arm["coordinates"][a], ship["coordinates"][b],
                                   atol=1e-5, rtol=0)
        checked += 1
    assert checked > 0


def test_neutral_arm_has_more_atoms_than_control():
    """The neutral arm carries the added nitrates; the control must not."""
    if not (_asset("neutral").exists() and _asset("control").exists()):
        pytest.skip("both arms not built")
    n = np.load(_asset("neutral")); c = np.load(_asset("control"))
    n_nodes = np.diff(n["node_ptr"]); c_nodes = np.diff(c["node_ptr"])
    assert len(n_nodes) == len(c_nodes)
    diff = n_nodes - c_nodes
    # every added nitrate is 4 atoms, and n_add is 1..3
    assert (diff > 0).all(), "some neutral complex gained no atoms"
    assert set(np.unique(diff)).issubset({4, 8, 12}), (
        f"atom-count differences {sorted(set(np.unique(diff)))} are not whole "
        f"nitrates")
