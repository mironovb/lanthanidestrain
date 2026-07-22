#!/usr/bin/env python3
"""Guards for the persistence-image sweep.

Each test targets a specific way this sweep could produce a number that looks
like a result and is not one.  Two earlier bugs in this study -- a Kabsch
correspondence error and a donor-count rule -- were caught by exactly this kind
of check rather than by inspection, which is why they are written before the
fits rather than after.

The tests that need the diagram cache skip cleanly when it is absent, so the
suite is runnable before the cache job lands.  ``data/`` is only ever read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from automl.qc import pi_sweep_render as R
from automl.qc.pi_sweep_build import CACHE, _diagram_dims
from automl.topo import pi_split

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "data/processed/feature_blocks/complex_gfn2xtb_pi_images.npz"


def _cache():
    if not CACHE.exists():
        pytest.skip("diagram cache not built yet")
    with np.load(CACHE) as z:
        return {"points": z["points"], "dims": z["dims"], "ptr": z["ptr"],
                "build_ids": [str(b) for b in z["build_ids"]]}


# ---------------------------------------------------------------------------
# The rendering must be the shipped rendering
# ---------------------------------------------------------------------------

def test_renderer_matches_shipped_rasteriser():
    """The vectorised renderer is the shipped ``persistence_image``.

    The sweep's whole claim is that it retunes the *published* representation.
    If the rasteriser differs at the shipped settings then the sweep is
    exploring a different object and no comparison to the published P0 arm is
    meaningful.
    """
    from src.geometry_features import persistence_image, PI_RESOLUTION
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(5, 400))
        b = rng.uniform(0.0, 2.4, n)
        d = b + rng.uniform(0.01, 1.0, n)
        diagram = np.stack([b, d], axis=1)
        # Compared at float32, the precision both the shipped asset and the
        # rendered sweep images are stored at.  ``persistence_image`` already
        # casts to float32 while ``_render_one`` accumulates in float64, so a
        # float64 comparison would only be measuring that cast (~1e-7 relative).
        mine = R._render_one(diagram, R.SHIPPED_CONFIG).astype(np.float32)
        ref = persistence_image(diagram, resolution=PI_RESOLUTION)
        scale = max(float(np.abs(ref).max()), 1e-12)
        assert np.abs(mine - ref).max() / scale < 1e-6


def test_render_respects_the_admission_window():
    """Points outside the birth/death window contribute nothing.

    The shipped window discards 13.5 % of real points; widening it is one of the
    swept axes, so the exclusion rule has to be exactly the shipped one or the
    "wider range" arms would differ for the wrong reason.
    """
    cfg = R.SHIPPED_CONFIG
    inside = np.array([[0.5, 1.0]])
    outside = np.array([[0.5, 1.0], [0.5, 9.9], [3.0, 4.0], [1.0, 0.5]])
    assert np.abs(R._render_one(inside, cfg) - R._render_one(outside, cfg)).max() < 1e-12

    wide = R.PIConfig(resolution=cfg.resolution, spread=cfg.spread, hi=6.0)
    assert R._render_one(np.array([[3.0, 4.0]]), wide).sum() > 0, \
        "a wider window must actually admit the points it was widened for"


@pytest.mark.parametrize("kind", R.WEIGHTS)
def test_weight_functions_are_nonnegative_and_monotone(kind):
    pers = np.linspace(0.01, 3.0, 50)
    w = R.weight_fn(kind, pers)
    assert (w >= 0).all()
    assert np.all(np.diff(w) >= -1e-12), f"{kind} is not non-decreasing"


def test_render_is_a_pure_function_of_the_diagram():
    """No target leakage: rendering depends on geometry alone.

    ``log D`` appears nowhere in this path.  Rendering the same diagram twice
    must be identical, and a permutation of the diagram's rows -- which carries
    no information -- must not change the image.
    """
    rng = np.random.default_rng(7)
    b = rng.uniform(0, 2.0, 200)
    diagram = np.stack([b, b + rng.uniform(0.05, 0.4, 200)], axis=1)
    a = R._render_one(diagram, R.SHIPPED_CONFIG)
    assert np.array_equal(a, R._render_one(diagram, R.SHIPPED_CONFIG))
    perm = rng.permutation(len(diagram))
    assert np.abs(a - R._render_one(diagram[perm], R.SHIPPED_CONFIG)).max() < 1e-12


def test_config_keys_are_injective():
    """Distinct configurations must not collide onto one images file.

    A collision would silently train two grid cells on the same images and
    report them as independent evidence.
    """
    cfgs = R.stage_a_configs()
    keys = [c.key() for c in cfgs]
    assert len(set(keys)) == len(cfgs)
    assert R.PIConfig(resolution=20).key() != R.PIConfig(resolution=32).key()
    assert R.PIConfig(spread=0.08).key() != R.PIConfig(spread=0.16).key()
    assert R.PIConfig(channels="sum").key() != R.PIConfig(channels="split").key()


def test_stage_a_covers_the_shipped_setting_and_smoother_ones():
    """The grid must contain the diagnosis it was designed around.

    The shipped spread is 0.61 of a pixel, so every Stage A cell is at least as
    smooth; the grid has to bracket that rather than sit entirely on one side.
    """
    cfgs = R.stage_a_configs()
    swept = [c for c in cfgs if c != R.SHIPPED_CONFIG]
    ratios = sorted({round(c.spread / c.pixel, 3) for c in swept})
    assert ratios == [0.5, 1.0, 2.0, 4.0]
    anchor = round(R.SHIPPED_CONFIG.spread / R.SHIPPED_CONFIG.pixel, 2)
    assert anchor == 0.61, f"shipped spread is {anchor} pixels, not 0.61"
    assert min(ratios) < anchor < max(ratios), \
        "the shipped 0.61-pixel spread is not bracketed by the swept grid"
    assert {c.resolution for c in swept} == set(R.STAGE_A_RESOLUTIONS)


def test_split_channels_partition_the_summed_image():
    """H0+H1 summed must equal the two split channels added back together.

    Otherwise "separate channels" would be changing the information content
    rather than only its layout, and a gain could not be attributed to the
    layout.
    """
    rng = np.random.default_rng(3)
    pts = np.stack([rng.uniform(0, 2.0, 300),
                    rng.uniform(0, 2.0, 300)], axis=1)
    pts[:, 1] = pts[:, 0] + rng.uniform(0.02, 0.4, 300)
    dims = rng.integers(0, 2, 300).astype(np.int8)
    cache = {"points": pts, "dims": dims,
             "ptr": np.array([0, 300]), "build_ids": ["x"]}
    summed, _ = R.render_all(cache, R.PIConfig(channels="sum"))
    split, _ = R.render_all(cache, R.PIConfig(channels="split"))
    assert summed.shape[1] == 1 and split.shape[1] == 2
    assert np.abs(summed[0, 0] - split[0].sum(axis=0)).max() < 1e-5


# ---------------------------------------------------------------------------
# The split that protects the endpoint
# ---------------------------------------------------------------------------

def test_split_is_a_partition():
    rec = pi_split.load()
    tune, confirm = set(rec["tune"]), set(rec["confirm"])
    assert not (tune & confirm), "tune and confirm overlap"
    assert len(tune) + len(confirm) == rec["n_extractants"] == 162
    assert rec["tune_pairs"] + rec["confirm_pairs"] == rec["n_pairs_total"] == 905


def test_split_is_reproducible_and_frozen():
    """Recomputing the rule must give back the frozen hash.

    If the split could drift between the sweep and the confirmation, the
    "untouched half" guarantee would be void and the endpoint would silently
    inherit the sweep's selection bias.
    """
    rec = pi_split.load()
    assert pi_split.build()["sha256"] == rec["sha256"]
    assert pi_split.digest(rec["tune"], rec["confirm"]) == rec["sha256"]


def test_split_balances_pairs_not_extractants():
    """Precision comes from pairs, so the halves are balanced on pairs.

    Balancing extractant *counts* instead would be nearly useless here: the top
    five extractants carry about 36 % of all adjacent pairs.
    """
    rec = pi_split.load()
    share = rec["confirm_pairs"] / rec["n_pairs_total"]
    assert 0.45 <= share <= 0.55, f"confirm half holds {share:.2%} of pairs"


def test_confirm_half_can_detect_a_known_effect():
    """The positive control, as a test.

    A confirm half that cannot see the already-published S0 effect would report
    a null for the persistence-image arm regardless of what the sweep found.
    That is the failure mode that made the 2/3-1/3 split unusable.
    """
    csv = pi_split.OUT_DIR / "positive_control.csv"
    if not csv.exists():
        pytest.skip("positive control not yet run")
    import pandas as pd
    row = pd.read_csv(csv).set_index("scope").loc["CONFIRM half"]
    assert bool(row["clears_zero"]), (
        f"confirm half does not detect S0: {row['delta']:+.4f} "
        f"[{row['lo']:+.4f}, {row['hi']:+.4f}]")


# ---------------------------------------------------------------------------
# The cache itself
# ---------------------------------------------------------------------------

def test_cache_agrees_with_the_shipped_diagram_function():
    """Keeping the homology dimension must not have changed anything else."""
    from src.geometry_features import persistence_diagram, PI_HOMOLOGY_DIMS, read_extxyz
    from automl.qc.pi_sweep_build import shipped_build_ids, geometry_paths
    _cache()                                   # skip if the cache is absent
    paths = geometry_paths()
    for bid in shipped_build_ids()[:3]:
        g = read_extxyz(Path(paths[bid]))
        pts, dims = _diagram_dims(g.coordinates)
        assert np.array_equal(pts[np.isin(dims, PI_HOMOLOGY_DIMS)],
                              persistence_diagram(g.coordinates))


def test_cache_covers_every_shipped_complex():
    cache = _cache()
    with np.load(SHIPPED) as z:
        shipped_ids = [str(b) for b in z["build_ids"]]
    assert set(shipped_ids) <= set(cache["build_ids"])
    assert cache["ptr"][0] == 0
    assert cache["ptr"][-1] == len(cache["points"])
    assert len(cache["dims"]) == len(cache["points"])


def test_cache_reproduces_the_shipped_images():
    """The gate, as a test: shipped settings in, shipped asset out.

    Not all of it, though.  18 of the 953 complexes cannot be reproduced in this
    environment by *any* code path -- the shipped asset was built under a
    different gudhi/CGAL, and calling the shipped functions directly reproduces
    the same mismatch.  So the assertion is that the overwhelming majority match
    exactly, and ``test_mismatches_are_environmental`` carries the attribution.
    """
    cache = _cache()
    imgs, ids = R.render_all(cache, R.SHIPPED_CONFIG)
    with np.load(SHIPPED) as z:
        ref = z["images"].astype(np.float32)
        ref_ids = [str(b) for b in z["build_ids"]]
    pos = {b: i for i, b in enumerate(ids)}
    got = imgs[[pos[b] for b in ref_ids]]
    assert got.shape == ref.shape
    n = len(ref_ids)
    rel = (np.abs(got - ref).reshape(n, -1).max(axis=1) /
           np.maximum(np.abs(ref).reshape(n, -1).max(axis=1), 1e-12))
    assert (rel <= 1e-5).mean() > 0.95, \
        f"only {(rel <= 1e-5).mean():.1%} of complexes reproduce exactly"


def test_mismatches_are_environmental_not_ours():
    """Every mismatch must be reproduced by the shipped functions themselves.

    This is the load-bearing half of the gate.  A complex that disagrees with
    this renderer but agrees with ``persistence_image(persistence_diagram(...))``
    would mean the cache is wrong, and every sweep number would be measuring
    something other than a retuned version of the published representation.
    """
    from src.geometry_features import (read_extxyz, persistence_diagram,
                                       persistence_image, PI_RESOLUTION)
    from automl.qc.pi_sweep_build import geometry_paths

    cache = _cache()
    imgs, ids = R.render_all(cache, R.SHIPPED_CONFIG)
    with np.load(SHIPPED) as z:
        ref = z["images"].astype(np.float32)
        ref_ids = [str(b) for b in z["build_ids"]]
    pos = {b: i for i, b in enumerate(ids)}
    got = imgs[[pos[b] for b in ref_ids]]
    n = len(ref_ids)
    scale = np.maximum(np.abs(ref).reshape(n, -1).max(axis=1), 1e-12)
    rel = np.abs(got - ref).reshape(n, -1).max(axis=1) / scale

    paths = geometry_paths()
    for i in np.where(rel > 1e-5)[0]:
        g = read_extxyz(Path(paths[ref_ids[i]]))
        direct = persistence_image(persistence_diagram(g.coordinates),
                                   resolution=PI_RESOLUTION)
        r_direct = float(np.abs(direct - ref[i, 0]).max()) / scale[i]
        assert abs(r_direct - rel[i]) < 1e-6, (
            f"{ref_ids[i]}: shipped functions deviate {r_direct:.3e} but this "
            f"renderer deviates {rel[i]:.3e} -- the cache is at fault")


def test_shipped_anchor_is_in_the_sweep():
    """The untuned comparison point must be rendered from the same cache.

    Because the shipped asset is not bit-reproducible here, comparing a tuned
    configuration against it would confound tuning with the environment
    discrepancy.  Sweeping the shipped configuration itself removes that.
    """
    assert R.SHIPPED_CONFIG in R.stage_a_configs()


# ---------------------------------------------------------------------------
# The readout changes are backward compatible
# ---------------------------------------------------------------------------

def test_single_channel_normalisation_is_unchanged():
    """Per-channel standardisation must be a no-op for the shipped images.

    Published PI-CNN runs have to remain reproducible; if this changed their
    inputs, the P0 numbers quoted throughout the study would no longer be the
    numbers the code produces.
    """
    from automl.topo.pi_cnn import PersistenceImages
    P = PersistenceImages(SHIPPED)
    assert P.n_channels == 1
    with np.load(SHIPPED) as z:
        raw = np.log1p(z["images"].astype(np.float32))
    assert np.abs(P.images - raw / raw.std()).max() < 1e-6


def test_every_cache_accepts_the_conformer_argument():
    """All three caches must share ComplexCache's signature.

    The training loop calls ``cache.batch(ids, confs)`` unconditionally, passing
    ``None`` when there is one conformer.  ``--conformers`` widened
    ``ComplexCache`` but not ``ImageCache`` or ``NullCache``, so ``--arch picnn``
    and ``--arch tabular`` raised TypeError from 33324ea onwards.  Nothing
    re-ran those arms until this sweep, so it stayed silent for two days -- and
    the published P0 and T0w numbers predate the regression, meaning the
    committed code could not have reproduced them.

    A signature check is cheap; re-running a GPU array to discover it is not.
    """
    import inspect
    from automl.topo.train import ImageCache, ComplexCache
    from automl.topo.tabular_net import NullCache
    for cls in (ComplexCache, ImageCache, NullCache):
        params = list(inspect.signature(cls.batch).parameters)
        assert params[:3] == ["self", "ids", "conformers"], \
            f"{cls.__name__}.batch has signature {params}"
        assert (inspect.signature(cls.batch).parameters["conformers"].default
                is None), f"{cls.__name__}.batch conformers is not optional"


def test_cnn_accepts_both_channel_counts():
    import torch
    from automl.topo.pi_cnn import PersistenceCNN
    for c, res in ((1, 20), (2, 48)):
        m = PersistenceCNN(dim=32, tabular_dim=0, in_channels=c)
        out = m({"image": torch.zeros(4, c, res, res)})
        assert out.shape == (4,)
