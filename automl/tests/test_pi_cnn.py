#!/usr/bin/env python3
"""Correctness tests for the persistence-image CNN readout.

The point of this module is to correct an invalid earlier test (the images were
flattened into a tabular model, against the asset's own stated contract), so
these tests check the things that make the correction real: that the image
arrives with its 2D structure intact, that the CNN actually uses spatial
adjacency, and that the tabular block is fused rather than ignored.

Run:  python3 -m pytest automl/tests/test_pi_cnn.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from automl.topo.pi_cnn import PersistenceImages, PersistenceCNN  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def images():
    return PersistenceImages()


def test_images_keep_their_2d_shape(images):
    assert images.images.ndim == 4, "images must stay (N, C, H, W), not flattened"
    assert images.images.shape[1:] == (1, 20, 20)
    assert np.isfinite(images.images).all()


def test_manifest_contract_is_the_one_we_are_honouring():
    """The readout contract this module exists to satisfy."""
    man = json.loads(
        (REPO / "data/processed/feature_blocks/feature_blocks_manifest.json").read_text())
    blob = json.dumps(man)
    assert "do_not_flatten_into_tabular_MLP" in blob, \
        "manifest contract changed -- revisit why pi_cnn.py exists"


def test_build_ids_are_unique_and_resolvable(images):
    assert len(set(images.build_ids)) == len(images.build_ids)
    assert images.index_of(images.build_ids[0]) == 0
    assert images.index_of("no-such-build-id") is None


def test_batch_selects_the_requested_images(images):
    b = images.batch([3, 1, 7], device="cpu")
    assert b["image"].shape == (3, 1, 20, 20)
    assert b["n_complexes"] == 3
    assert torch.allclose(b["image"][1], torch.as_tensor(images.images[1]))


def _model(tabular_dim=0, seed=0):
    torch.manual_seed(seed)
    m = PersistenceCNN(dim=16, dropout=0.0, tabular_dim=tabular_dim,
                       channels=(8, 16))
    m.eval()
    return m


def test_cnn_uses_spatial_structure(images):
    """A permuted image must change the prediction.

    This is the test the flattened version would fail by construction: a tabular
    model over 400 independent pixels is invariant to a fixed permutation of
    them, so it cannot be using birth-death adjacency at all.
    """
    x = torch.as_tensor(images.images[:1]).clone()
    rng = np.random.default_rng(0)
    perm = torch.as_tensor(rng.permutation(400))
    x_shuf = x.reshape(1, 1, 400)[:, :, perm].reshape(1, 1, 20, 20)

    m = _model()
    with torch.no_grad():
        a = m({"image": x, "n_complexes": 1})
        b = m({"image": x_shuf, "n_complexes": 1})
    assert not torch.allclose(a, b, atol=1e-6), \
        "CNN output ignores pixel layout -- it is not using the image structure"


def test_batch_independence(images):
    x = torch.as_tensor(images.images[:3])
    m = _model()
    with torch.no_grad():
        alone = m({"image": x[:1], "n_complexes": 1})
        together = m({"image": x, "n_complexes": 3})
    assert torch.allclose(alone, together[:1], atol=1e-5), \
        "prediction leaks across the batch"


def test_hybrid_head_consumes_tabular_features(images):
    x = torch.as_tensor(images.images[:1])
    m = _model(tabular_dim=8)
    tab = torch.zeros(1, 8)
    with torch.no_grad():
        a = m({"image": x, "n_complexes": 1}, tabular=tab)
        b = m({"image": x, "n_complexes": 1}, tabular=tab + 1.0)
    assert not torch.allclose(a, b), "tabular block is being ignored"


def test_encoder_is_small_enough_for_953_images():
    m = PersistenceCNN(dim=64, tabular_dim=746)
    n_enc = sum(p.numel() for n, p in m.named_parameters()
                if n.startswith(("conv", "proj")))
    assert n_enc < 200_000, f"image encoder has {n_enc} params for 953 images"
