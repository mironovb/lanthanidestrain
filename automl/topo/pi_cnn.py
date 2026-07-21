#!/usr/bin/env python3
"""CNN readout for GFN2-xTB persistence images.

Why this exists
---------------
``feature_blocks_manifest.json`` ships the persistence images with an explicit
contract:

    "readout": "CNN_or_ViT; do_not_flatten_into_tabular_MLP"

The earlier study violated that contract -- block ``g11`` flattened the 20x20
image into 279 tabular columns and fed it to a gradient-boosted tree, which
measured ``DR2 = +0.004`` and degraded the selectivity metrics.  That test was
invalid by the asset's own terms: flattening destroys the spatial adjacency in
the birth-death plane that makes a persistence image an image at all, and a
tree then has to rediscover locality from 279 independent columns with 953
distinct examples.  This module gives the images the readout they were built
for, so the persistence-homology claim gets a fair test.

Design
------
Persistence images are small (20x20), smooth, non-negative and sparse, and
there are only 953 of them.  The encoder is deliberately tiny -- three conv
blocks, ~35k parameters -- because anything larger memorises 953 images long
before it generalises across extractants.  Same hybrid-head pattern as the
simplicial network: the image embedding is concatenated with the tabular block,
since the claim under test is that topology *augments* 2D descriptors.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
# PI_IMAGES_PATH lets a run point at images rebuilt from a different geometry
# set (e.g. the tight, ALPB-solvated re-optimisations) without editing code or
# touching data/.  The rebuild uses identical functions and constants, verified
# to reproduce the shipped asset bit-for-bit, so a difference downstream is
# attributable to the geometry alone.
PI_PATH = Path(os.environ.get(
    "PI_IMAGES_PATH",
    REPO / "data/processed/feature_blocks/complex_gfn2xtb_pi_images.npz"))


class PersistenceImages:
    """The shipped 953 x 1 x 20 x 20 persistence images, keyed by build id."""

    def __init__(self, path: Path | None = None):
        path = Path(path) if path is not None else PI_PATH
        with np.load(path) as z:
            self.images = z["images"].astype(np.float32)
            self.build_ids = [str(b) for b in z["build_ids"]]
        self._index = {b: i for i, b in enumerate(self.build_ids)}
        # Persistence images are non-negative with a very long right tail (a few
        # pixels near the diagonal carry most of the mass).  A log1p keeps the
        # dynamic range trainable without destroying the ordering.
        self.images = np.log1p(self.images)
        s = self.images.std()
        self.images = self.images / (s if s > 1e-8 else 1.0)

    def __len__(self) -> int: return len(self.build_ids)

    def index_of(self, build_id: str) -> int | None:
        return self._index.get(str(build_id))

    def batch(self, ids: list[int], device) -> dict:
        x = torch.as_tensor(self.images[ids]).to(device)
        return {"image": x, "n_complexes": len(ids)}


class PersistenceCNN(nn.Module):
    """Small CNN over persistence images, interface-compatible with SimplicialNet."""

    def __init__(self, dim: int = 64, dropout: float = 0.15,
                 tabular_dim: int = 0, head_hidden: int = 256,
                 channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        c_in = 1
        blocks = []
        for c_out in channels:
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1),
                       nn.BatchNorm2d(c_out), nn.SiLU()]
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        # mean + max pooling over the birth-death plane
        self.proj = nn.Linear(2 * c_in, dim)
        self.dim = dim
        self.embed_dim = dim
        self.tabular_dim = tabular_dim
        head_in = dim + tabular_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1))

    def encode(self, batch: dict) -> torch.Tensor:
        h = self.conv(batch["image"])                      # (B, C, 20, 20)
        mean = h.mean(dim=(2, 3))
        mx = h.amax(dim=(2, 3))
        return self.proj(torch.cat([mean, mx], dim=-1))

    def forward(self, batch: dict, tabular: torch.Tensor | None = None):
        emb = self.encode(batch)
        if self.tabular_dim:
            if tabular is None:
                raise ValueError("model built with tabular_dim>0 but none given")
            emb = torch.cat([emb, tabular], dim=-1)
        return self.head(emb).squeeze(-1)
