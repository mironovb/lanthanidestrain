#!/usr/bin/env python3
"""Message-passing simplicial network over Ln(III)-complex Vietoris-Rips complexes.

Architecture
------------
Three simplex levels carry hidden states: nodes (0-simplices), edges
(1-simplices) and triangles (2-simplices).  Each layer exchanges messages
along the boundary maps in both directions:

    node   <- edge        (co-boundary: which edges contain this atom)
    edge   <- node        (boundary: the two atoms spanning this edge)
    edge   <- triangle    (co-boundary: which faces contain this edge)
    triangle <- edge      (boundary: the three edges bounding this face)

This is the Bodnar-style MPSN reduced to the messages the shipped asset can
support exactly.  Upper/lower *adjacency* between same-level simplices is
deliberately omitted: on a complex with ~10^4 triangles the adjacency lists are
an order of magnitude larger than the boundary maps and buy nothing that two
rounds of boundary/co-boundary passing does not already reach.

Invariance, by construction rather than by augmentation
-------------------------------------------------------
No raw coordinate ever enters the network.  Node inputs are the xTB partial
charge, the metal/donor flags and the distance to the metal; edge and triangle
inputs are their VR filtration radii.  All of these are invariant to rotation,
translation and reflection, so the model is exactly invariant -- there is
nothing for a rotation-augmentation to teach it.  Permutation invariance
follows from scatter-mean/max aggregation.  Both are asserted in the tests.

Metal-centred readout
---------------------
Pooling over a 300-atom complex dilutes the coordination sphere, which is the
part that decides selectivity.  The readout therefore concatenates the global
pooled state with the **metal node's own embedding** and the pooled states of
the simplices incident to it.

Hybrid head
-----------
The claim under test is that topology *augments* 2D descriptors, so the default
forward pass concatenates the simplicial embedding with the tabular feature
vector (ECFP + RDKit + conditions + metal) before the regression head.  Passing
``tabular_dim=0`` gives the topology-only ablation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    total = scatter_sum(src, index, dim_size)
    count = src.new_zeros(dim_size).index_add_(
        0, index, src.new_ones(src.shape[0]))
    return total / count.clamp(min=1.0).unsqueeze(-1)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_full((dim_size, src.shape[-1]), float("-inf"))
    out = out.index_reduce_(0, index, src, "amax", include_self=True)
    return torch.nan_to_num(out, neginf=0.0)


def _mlp(din: int, dhid: int, dout: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dhid), nn.SiLU(), nn.Dropout(dropout),
        nn.Linear(dhid, dout))


class MPSNLayer(nn.Module):
    """One round of boundary / co-boundary message passing across three levels."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_e_from_n = _mlp(2 * dim, dim, dim, dropout)   # nodes -> edge
        self.msg_n_from_e = _mlp(2 * dim, dim, dim, dropout)   # edge  -> node
        self.msg_t_from_e = _mlp(2 * dim, dim, dim, dropout)   # edges -> triangle
        self.msg_e_from_t = _mlp(2 * dim, dim, dim, dropout)   # triangle -> edge
        self.upd_n = _mlp(2 * dim, dim, dim, dropout)
        self.upd_e = _mlp(3 * dim, dim, dim, dropout)
        self.upd_t = _mlp(2 * dim, dim, dim, dropout)
        self.norm_n, self.norm_e, self.norm_t = (nn.LayerNorm(dim),
                                                 nn.LayerNorm(dim),
                                                 nn.LayerNorm(dim))

    def forward(self, hn, he, ht, edge_index, tri_edges):
        n_nodes, n_edges, n_tris = hn.shape[0], he.shape[0], ht.shape[0]
        i, j = edge_index[0], edge_index[1]

        # --- edge <- its two endpoint nodes (symmetrised: an edge is unordered)
        pair = torch.cat([hn[i] + hn[j], (hn[i] - hn[j]).abs()], dim=-1)
        m_e_n = self.msg_e_from_n(pair)

        # --- node <- incident edges
        # One mean over ALL incident edges, not the sum of two per-endpoint
        # means.  An edge is unordered, so whether a given node sits in row i
        # or row j of edge_index is arbitrary; summing two separately
        # normalised means makes the result depend on that arbitrary split and
        # silently breaks permutation invariance (verified: the two orderings
        # differed by 2.6e-4 identically in float32 and float64, so it was not
        # rounding).  Concatenating first normalises by true node degree.
        em = self.msg_n_from_e(torch.cat([he, he], dim=-1))
        m_n_e = scatter_mean(torch.cat([em, em], dim=0),
                             torch.cat([i, j], dim=0), n_nodes)

        # --- triangle <- its three edges (sum is permutation invariant)
        if n_tris > 0:
            e0, e1, e2 = tri_edges[0], tri_edges[1], tri_edges[2]
            tri_in = he[e0] + he[e1] + he[e2]
            m_t_e = self.msg_t_from_e(torch.cat([tri_in, ht], dim=-1))
            # --- edge <- incident triangles
            tm = self.msg_e_from_t(torch.cat([ht, ht], dim=-1))
            # Same argument as above: which of a triangle's three bounding
            # edges lands in slot e0/e1/e2 is arbitrary, so the mean must be
            # taken once over all incident triangles.
            m_e_t = scatter_mean(torch.cat([tm, tm, tm], dim=0),
                                 torch.cat([e0, e1, e2], dim=0), n_edges)
        else:
            m_t_e = ht.new_zeros((0, ht.shape[-1]))
            m_e_t = he.new_zeros((n_edges, he.shape[-1]))

        hn = self.norm_n(hn + self.upd_n(torch.cat([hn, m_n_e], dim=-1)))
        he = self.norm_e(he + self.upd_e(torch.cat([he, m_e_n, m_e_t], dim=-1)))
        if n_tris > 0:
            ht = self.norm_t(ht + self.upd_t(torch.cat([ht, m_t_e], dim=-1)))
        return hn, he, ht


class SimplicialNet(nn.Module):
    """MPSN encoder + optional tabular fusion + regression head."""

    def __init__(self, dim: int = 96, layers: int = 3, dropout: float = 0.1,
                 tabular_dim: int = 0, head_hidden: int = 256,
                 n_z: int = 32, node_feat_dim: int = 5,
                 radial_bins: int = 32, radial_max: float = 8.0,
                 head_embed_mult: int = 1):
        super().__init__()
        self.dim = dim
        self.z_emb = nn.Embedding(n_z, dim)
        self.node_in = _mlp(node_feat_dim, dim, dim, dropout)
        self.edge_in = _mlp(1, dim, dim, dropout)
        self.tri_in = _mlp(1, dim, dim, dropout)
        self.layers = nn.ModuleList(
            [MPSNLayer(dim, dropout) for _ in range(layers)])

        # Explicit metal-centred radial readout.
        #
        # Motivated by a measurement, not by taste: with the metal element
        # masked, eight hand-made M-L distance summaries recover the lanthanide
        # index at R2 = 0.57 under leave-extractants-out CV, while this encoder
        # without the radial term managed R2 = 0.016 on the same folds.  The
        # lanthanide contraction lives in sub-0.1 A shifts of the coordination
        # shell, and mean/max pooling over ~100 node embeddings averages exactly
        # that away.  A soft histogram of distance-to-metal keeps it: each node
        # contributes a Gaussian bump at its own radius, so the *shape* of the
        # coordination shell survives pooling.
        #
        # Still exactly invariant: distance-to-metal is a scalar under rotation,
        # translation and reflection, and the sum over nodes is permutation
        # invariant.
        self.radial_bins = radial_bins
        centres = torch.linspace(0.0, radial_max, radial_bins)
        self.register_buffer("radial_centres", centres)
        # width ~ one bin spacing, so neighbouring bins overlap smoothly
        self.radial_width = float(radial_max / max(radial_bins - 1, 1))
        self.radial_proj = _mlp(2 * radial_bins, dim, dim, dropout)

        # global mean+max over 3 levels (6) + metal node (1) + metal-incident
        # edge mean (1) + radial shell (1) = 9 blocks of `dim`
        self.embed_dim = 9 * dim
        self.tabular_dim = tabular_dim
        # ``head_embed_mult`` widens only the *head*, not ``encode``.  The
        # block-centred arm concatenates each embedding with its deviation from
        # the composition-block mean, which happens in the training loop where
        # block membership is known; the encoder itself is unchanged, so a model
        # built with mult=1 stays byte-compatible with every existing run.
        self.head_embed_mult = int(head_embed_mult)
        head_in = self.head_embed_mult * self.embed_dim + tabular_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1))

    # -- encoder ------------------------------------------------------------
    def encode(self, batch: dict[str, Any]) -> torch.Tensor:
        hn = self.z_emb(batch["z_idx"]) + self.node_in(batch["node_feat"])
        he = self.edge_in(batch["edge_filt"])
        ht = self.tri_in(batch["tri_filt"])
        ei, te = batch["edge_index"], batch["tri_edges"]
        for layer in self.layers:
            hn, he, ht = layer(hn, he, ht, ei, te)

        nb, eb, tb = batch["node_batch"], batch["edge_batch"], batch["tri_batch"]
        B = int(batch["n_complexes"])
        parts = [scatter_mean(hn, nb, B), scatter_max(hn, nb, B),
                 scatter_mean(he, eb, B), scatter_max(he, eb, B)]
        if ht.shape[0] > 0:
            parts += [scatter_mean(ht, tb, B), scatter_max(ht, tb, B)]
        else:
            parts += [hn.new_zeros((B, self.dim)), hn.new_zeros((B, self.dim))]

        # metal-centred readout
        midx = batch["metal_index"]
        parts.append(hn[midx])
        is_metal_edge = (ei[0].unsqueeze(0) == midx.unsqueeze(1)) | \
                        (ei[1].unsqueeze(0) == midx.unsqueeze(1))   # (B, E)
        w = is_metal_edge.to(he.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        parts.append((w @ he) / denom)

        # radial shell profile: soft histogram of distance-to-metal, computed
        # separately over all atoms and over coordinating donors only
        d = batch["node_feat"][:, 4]                       # dist_to_metal
        is_donor = batch["node_feat"][:, 3]
        rbf = torch.exp(-((d.unsqueeze(-1) - self.radial_centres) ** 2)
                        / (2.0 * self.radial_width ** 2))   # (N, bins)
        all_hist = scatter_sum(rbf, nb, B)
        don_hist = scatter_sum(rbf * is_donor.unsqueeze(-1), nb, B)
        parts.append(self.radial_proj(torch.cat([all_hist, don_hist], dim=-1)))
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict[str, Any],
                tabular: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.encode(batch)
        if self.tabular_dim:
            if tabular is None:
                raise ValueError("model built with tabular_dim>0 but none given")
            emb = torch.cat([emb, tabular], dim=-1)
        return self.head(emb).squeeze(-1)


class MaskedChargeHead(nn.Module):
    """Self-supervised pretraining head.

    Only 953 distinct structures back 4,746 rows, so the supervised signal is
    thin.  Pretraining asks the encoder to reconstruct masked xTB partial
    charges and edge filtration radii from topology alone -- targets that are
    free, defined on every complex, and force the encoder to represent local
    chemical environment before it ever sees a log D.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.charge = nn.Linear(dim, 1)
        self.filt = nn.Linear(dim, 1)

    def forward(self, hn: torch.Tensor, he: torch.Tensor):
        return self.charge(hn).squeeze(-1), self.filt(he).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
