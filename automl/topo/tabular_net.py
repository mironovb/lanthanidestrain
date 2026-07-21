#!/usr/bin/env python3
"""The missing control: everything the topological arms do, minus the topology.

Why this file exists
--------------------
All 51 runs of the topological study used a topological encoder.  The mechanism
those runs identified was *"train the contrast, not the absolute value"* -- an
auxiliary loss on within-composition pairwise differences plus checkpoint
selection on adjacent-pair R2.  Both of those are properties of the **training
objective**, not of the representation, so a tabular model given the same
objective might capture most of the +0.243 gain.  Until that is measured, the
gain cannot be attributed to topology at all.

``TabularNet`` is that measurement.  It is deliberately not a new baseline: it
is the same training loop, the same folds, the same contrast loss, the same
adjacent-pair selection and the same 16 seeds, with the topological embedding
removed.  Because ``run_fold`` composes the head input as
``cat([encode(batch), tabular])``, an encoder that returns a width-zero tensor
reduces that expression to the tabular block exactly -- so no other line of the
harness changes, and any difference in the result is attributable to the
embedding and nothing else.

Zero-width, not zeroed
----------------------
The same argument the ``--topology-only`` ablation already makes, mirrored.
Returning a tensor of zeros would leave ``head``'s first-layer weights for those
864 columns present and trainable against a constant input: a differently-sized
model fitting a degenerate feature, which is not the ablation being claimed.
Width zero means those weights do not exist.  The two ablations are then exact
complements and the 2x2 is honest.

A note on capacity, since it is the obvious objection
-----------------------------------------------------
This model has fewer parameters than the topological arms by construction --
the head loses its 864 embedding columns and the encoder is gone.  That
asymmetry is not neutral, so it is measured rather than argued: a wide variant
(``head_hidden=512``) is run alongside, and the control's reported value is the
**better** of the two.  Taking the max makes the primary test harder for
topology to pass, which is the direction a control should err in.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class NullCache:
    """Stands in for ComplexCache / ImageCache when there is no 3D asset.

    ``run_fold`` still batches by distinct complex and gathers per row, because
    that is what keeps the control's *sampling* identical to the topological
    arms.  Only the payload is empty.
    """

    def __init__(self, device):
        self.device = device

    def batch(self, ids: list[int]) -> dict[str, Any]:
        return {"n_complexes": len(ids)}


class TabularNet(nn.Module):
    """Tabular-only model, interface-compatible with SimplicialNet / PersistenceCNN.

    The head is character-for-character the head the other two arms use, so the
    comparison isolates the embedding rather than the regressor on top of it.
    """

    def __init__(self, dim: int = 96, dropout: float = 0.15,
                 tabular_dim: int = 0, head_hidden: int = 256):
        super().__init__()
        if tabular_dim <= 0:
            # Width-zero embedding AND width-zero tabular block is a model with
            # no inputs whatsoever.  Silently training it would produce a run
            # record that looks like a control but measures the target mean.
            raise ValueError(
                "TabularNet needs tabular features: tabular_dim must be > 0. "
                "--arch tabular with --topology-only leaves the model no inputs.")
        self.dim = dim
        self.embed_dim = 0
        self.tabular_dim = tabular_dim
        head_in = self.embed_dim + tabular_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1))

    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        """A (B, 0) tensor -- present so the harness needs no branch, empty so
        the model cannot see structure."""
        p = next(self.parameters())
        return torch.zeros((int(batch["n_complexes"]), 0),
                           device=p.device, dtype=p.dtype)

    def forward(self, batch: dict[str, Any],
                tabular: torch.Tensor | None = None) -> torch.Tensor:
        if tabular is None:
            raise ValueError("TabularNet requires the tabular block")
        emb = self.encode(batch)
        return self.head(torch.cat([emb.new_zeros((tabular.shape[0], 0)),
                                    tabular], dim=-1)).squeeze(-1)
