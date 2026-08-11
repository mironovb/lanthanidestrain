#!/usr/bin/env python3
"""A plain distance-based 3D network, with no simplicial structure at all.

The question this exists to answer
----------------------------------
It is the oldest unanswered one in the study, named in ``PI_EMAIL.md`` sec 9 as
next step 2, in ``SYNTHESIS.md`` as "the one boundary the filtration test could
not probe", and in ``STACK_RESULTS.md`` sec 8 as what would settle it:

    is "simplicial" the operative ingredient, or merely "3D message passing"?

The published claim is that message passing over a Vietoris-Rips complex carries
adjacent-lanthanide selectivity that fingerprints do not.  The filtration
replication (3.0 / 4.0 A) showed the effect is not a tuned radius, but every arm
that has ever shown it was simplicial.  A persistence-image CNN -- the only other
topological representation tried -- did not reproduce it, which leaves two live
readings that no existing run separates:

* topology, specifically, is doing the work; or
* *any* learned encoder over the 3D structure would do, and the simplicial
  complex is one arbitrary way to reach it.

This module is the second reading's champion.  It is a SchNet-style continuous
filter network: messages flow along the same edges, weighted by a learned
function of the interatomic distance, and there is no notion of a triangle, a
boundary map or a filtration anywhere in it.

What is held fixed, so the comparison means something
-----------------------------------------------------
Everything except the encoder body:

* the same edges, from the same Vietoris-Rips asset at the same cutoff -- so the
  neighbourhood definition is identical and only the *algebra over it* differs;
* the same node inputs (xTB partial charge, metal/donor flags, distance to
  metal) and the same 27-element Z vocabulary;
* the same metal-centred readout and the same radial shell histogram, copied
  from ``SimplicialNet`` deliberately rather than reinvented -- that readout was
  added because pooling over a 300-atom complex averages the coordination sphere
  away, and a control that omitted it would lose for a reason having nothing to
  do with simplicial structure;
* the same embedding width (9 blocks of ``dim``), so the head sees a vector of
  the same shape and no capacity difference is smuggled in.

Invariance is exact, by construction, for the same reason it is in ``snn.py``:
no raw coordinate enters, only distances and scalars, and all aggregation is
scatter-based.  The tests assert it.

Reading the outcome
-------------------
Pre-registered in ``ENCODER_PREREGISTRATION.md``.  Both outcomes are results:

* it adds to the stack  -> the claim broadens from "Vietoris-Rips message
  passing" to "learned 3D representations", and the paper's contribution is the
  *rule* for what a candidate representation must satisfy, not the complex;
* it does not add       -> the claim is bounded to simplicial message passing,
  which is a sharper and more surprising statement than the current one.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from automl.topo.snn import _mlp, scatter_max, scatter_mean, scatter_sum


class RBFExpansion(nn.Module):
    """Gaussian radial basis over interatomic distance.

    The edge input carried by the Vietoris-Rips asset is the filtration radius,
    which for a 1-simplex *is* the distance between its two atoms -- so no new
    geometry is read and no new asset is needed.
    """

    def __init__(self, n_bins: int = 32, r_max: float = 5.0):
        super().__init__()
        centres = torch.linspace(0.0, r_max, n_bins)
        self.register_buffer("centres", centres)
        self.width = float(r_max / max(n_bins - 1, 1))
        self.n_bins = n_bins

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        d = d.reshape(-1, 1)
        return torch.exp(-((d - self.centres) ** 2) / (2.0 * self.width ** 2))


class CFConvLayer(nn.Module):
    """One continuous-filter convolution, symmetrised over the edge direction.

    An edge in the shipped asset is unordered, so whether an atom sits in row i
    or row j of ``edge_index`` is arbitrary.  Messages are therefore passed both
    ways and aggregated once over all incident edges -- the same correction that
    ``MPSNLayer`` documents, made for the same reason: normalising two directions
    separately would make the result depend on that arbitrary split and silently
    break permutation invariance.
    """

    def __init__(self, dim: int, rbf_bins: int, dropout: float = 0.1):
        super().__init__()
        self.filter_net = _mlp(rbf_bins, dim, dim, dropout)
        self.pre = nn.Linear(dim, dim)
        self.post = _mlp(dim, dim, dim, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, hn: torch.Tensor, rbf: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        i, j = edge_index[0], edge_index[1]
        w = self.filter_net(rbf)                      # (E, dim)
        x = self.pre(hn)
        msg = torch.cat([x[j] * w, x[i] * w], dim=0)  # both directions
        dst = torch.cat([i, j], dim=0)
        agg = scatter_mean(msg, dst, hn.shape[0])
        return self.norm(hn + self.post(agg))


class DistanceNet(nn.Module):
    """Continuous-filter 3D encoder + tabular fusion + regression head.

    Signature matches ``SimplicialNet`` so ``train.run_fold`` can build either
    without a special case, and so the head, the block-centred arm and the
    embedding dump all behave identically.
    """

    def __init__(self, dim: int = 96, layers: int = 3, dropout: float = 0.1,
                 tabular_dim: int = 0, head_hidden: int = 256,
                 n_z: int = 32, node_feat_dim: int = 5,
                 radial_bins: int = 32, radial_max: float = 8.0,
                 head_embed_mult: int = 1, rbf_bins: int = 32,
                 rbf_max: float = 5.0, pair_head: bool = False,
                 film_dim: int = 0):
        super().__init__()
        self.dim = dim
        self.z_emb = nn.Embedding(n_z, dim)
        self.node_in = _mlp(node_feat_dim, dim, dim, dropout)
        self.rbf = RBFExpansion(rbf_bins, rbf_max)
        self.layers = nn.ModuleList(
            [CFConvLayer(dim, rbf_bins, dropout) for _ in range(layers)])

        # Readout copied from SimplicialNet, deliberately.  See module docstring:
        # the metal-centred and radial terms exist because mean/max pooling over
        # ~100 node embeddings averages away exactly the sub-0.1 A shell shifts
        # the lanthanide contraction lives in.  A control lacking them would lose
        # for a reason unrelated to simplicial structure.
        self.radial_bins = radial_bins
        centres = torch.linspace(0.0, radial_max, radial_bins)
        self.register_buffer("radial_centres", centres)
        self.radial_width = float(radial_max / max(radial_bins - 1, 1))
        self.radial_proj = _mlp(2 * radial_bins, dim, dim, dropout)
        self.edge_proj = _mlp(rbf_bins, dim, dim, dropout)

        # Same 9 blocks of `dim` as SimplicialNet, so the head sees an
        # identically shaped vector: node mean/max, edge mean/max, two blocks
        # standing in for the triangle levels this model does not have, the
        # metal node, its incident-edge mean, and the radial shell.
        self.embed_dim = 9 * dim
        self.tabular_dim = tabular_dim
        self.head_embed_mult = int(head_embed_mult)
        head_in = self.head_embed_mult * self.embed_dim + tabular_dim
        # CAMPAIGN3 T2, absent here until now.  The class docstring claimed the
        # signature matched SimplicialNet, but ``pair_head`` was never a
        # parameter -- so ``--pair-head`` on an --arch dist run was accepted,
        # recorded as pair_head=True in results.jsonl, and did NOTHING.
        # ``--pair-reconcile`` then also no-opped, because it is gated on
        # ``getattr(model, "pair_head", None) is not None``.  Every modern run
        # uses --arch dist, so the flag has never once been exercised; a
        # 24-cell campaign returned four arms agreeing to six decimal places,
        # which is what exposed it.  Identical construction to SimplicialNet so
        # the two architectures remain comparable.
        # CAMPAIGN3 T3, also absent here until now (same cause as pair_head:
        # DistanceNet is built by its own call and never received film_dim, so
        # --film was a silent no-op on --arch dist).  The 64 condition columns --
        # 45 diluents, 9 acids, concentrations, temperature -- otherwise enter
        # only AFTER pooling, so kerosene and nitrobenzene give byte-identical
        # structural embeddings.  Worth having because separations are measurably
        # condition-dependent: the same (composition, adjacent pair) measured in
        # different strict blocks reproduces to 0.1533 against a spread of 0.2236.
        self.film_dim = int(film_dim or 0)
        self.film = (nn.Sequential(
            nn.Linear(self.film_dim, dim), nn.SiLU(),
            nn.Linear(dim, 2 * self.embed_dim)) if self.film_dim else None)
        self.use_pair_head = bool(pair_head)
        self.pair_head = (nn.Sequential(
            nn.Linear(3 * head_in, head_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.SiLU(),
            nn.Linear(head_hidden // 2, 1)) if self.use_pair_head else None)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1))

    # -- encoder ------------------------------------------------------------
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        hn = self.z_emb(batch["z_idx"]) + self.node_in(batch["node_feat"])
        ei = batch["edge_index"]
        rbf = self.rbf(batch["edge_filt"])
        for layer in self.layers:
            hn = layer(hn, rbf, ei)
        he = self.edge_proj(rbf)

        nb, eb = batch["node_batch"], batch["edge_batch"]
        B = int(batch["n_complexes"])
        parts = [scatter_mean(hn, nb, B), scatter_max(hn, nb, B),
                 scatter_mean(he, eb, B), scatter_max(he, eb, B),
                 # the two slots SimplicialNet fills from the triangle level.
                 # Held at zero rather than dropped so the embedding width, and
                 # therefore the head, is identical between the two encoders.
                 hn.new_zeros((B, self.dim)), hn.new_zeros((B, self.dim))]

        midx = batch["metal_index"]
        parts.append(hn[midx])
        is_metal_edge = (ei[0].unsqueeze(0) == midx.unsqueeze(1)) | \
                        (ei[1].unsqueeze(0) == midx.unsqueeze(1))
        w = is_metal_edge.to(he.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        parts.append((w @ he) / denom)

        d = batch["node_feat"][:, 4]
        is_donor = batch["node_feat"][:, 3]
        shell = torch.exp(-((d.unsqueeze(-1) - self.radial_centres) ** 2)
                          / (2.0 * self.radial_width ** 2))
        parts.append(self.radial_proj(
            torch.cat([scatter_sum(shell, nb, B),
                       scatter_sum(shell * is_donor.unsqueeze(-1), nb, B)],
                      dim=-1)))
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict[str, Any],
                tabular: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.encode(batch)
        if self.tabular_dim:
            if tabular is None:
                raise ValueError("model built with tabular_dim>0 but none given")
            emb = torch.cat([emb, tabular], dim=-1)
        return self.head(emb).squeeze(-1)

    def pair_forward(self, emb: torch.Tensor, i: torch.Tensor,
                     j: torch.Tensor) -> torch.Tensor:
        """Predict the pair difference directly -- see SimplicialNet.pair_forward.

        Byte-for-byte the same construction, so a dist-vs-snn comparison of the
        pair head measures the encoder and not two different heads.  [h_i, h_j,
        h_i - h_j] is antisymmetric-friendly: the difference term flips sign
        with pair order, which is the symmetry the target has.
        """
        if self.pair_head is None:
            raise ValueError("model built without --pair-head")
        z = torch.cat([emb[i], emb[j], emb[i] - emb[j]], dim=-1)
        return self.pair_head(z).squeeze(-1)

    def modulate(self, e: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """FiLM the pooled embedding on the condition columns.

        Residual form (1 + gamma), identical to SimplicialNet: at initialisation
        the transform is close to the identity, so turning FiLM on does not throw
        away the representation the rest of the study is built on.
        """
        if self.film is None:
            raise ValueError("model built without --film")
        g, b = self.film(cond).chunk(2, dim=-1)
        return (1.0 + g) * e + b
