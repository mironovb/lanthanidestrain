#!/usr/bin/env python3
"""An attention-based 3D encoder: a transformer over the atoms of a complex.

Everything except the encoder body is held identical to ``DistanceNet`` so
the comparison measures the body and nothing else: the same node inputs
(Z embedding, xTB charge, metal/donor flags, distance to metal), the same
Vietoris-Rips edges for the edge terms of the readout, the same nine-block
readout with the metal-centred and radial-shell terms, the same head, the
same losses in ``train.run_fold``.

The body: ``layers`` rounds of multi-head self-attention in which every atom
attends to every other atom of its own complex, with the attention logits
biased by a learned function of the interatomic distance (a radial basis
over the pairwise distance, mapped to one scalar per head -- the spatial
encoding of Graphormer).  Padding across complexes is masked.

Two modes, recorded in the run stem:

* dense (default): all pairs, distance bias saturating past ``bias_max``.
  This is the actual transformer: long-range geometry that the <= 4 A
  message-passing encoders never see.
* ``sparse=True``: attention restricted to the same <= cutoff neighbourhood
  the message-passing encoders use.  Isolates "attention instead of message
  passing" from "longer range".

Coordinates are read from ``batch["coords"]``, which ``train.ComplexCache``
adds only when this architecture is selected, so every other arm's batch
stays byte-identical.  Invariance to rotation and translation is exact:
only pairwise distances enter.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from automl.topo.dist_gnn import RBFExpansion
from automl.topo.snn import _mlp, scatter_max, scatter_mean, scatter_sum


def dense_index(node_batch: torch.Tensor, n_complexes: int):
    """Per-node (complex, position-in-complex) for padding into (B, N, d).

    ``collate`` concatenates complexes in order, so node_batch is
    non-decreasing and positions are offsets from the complex's first node.
    """
    counts = torch.bincount(node_batch, minlength=n_complexes)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(node_batch.numel(), device=node_batch.device) \
        - starts[node_batch]
    return counts, pos


class DistanceBiasedAttention(nn.Module):
    """Pre-norm transformer block with a per-head distance bias."""

    def __init__(self, dim: int, n_heads: int, rbf_bins: int, dropout: float):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} not divisible by heads {n_heads}")
        self.h, self.dh = n_heads, dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.bias = nn.Linear(rbf_bins, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = _mlp(dim, 2 * dim, dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, rbf_pair: torch.Tensor,
                allowed: torch.Tensor) -> torch.Tensor:
        B, N, d = H.shape
        x = self.norm1(H)
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # (B, h, N, dh)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        logits = logits + self.bias(rbf_pair).permute(0, 3, 1, 2)
        # -1e9 rather than -inf: padding QUERY rows have no allowed key and
        # would otherwise softmax to NaN; their outputs are discarded anyway.
        logits = logits.masked_fill(~allowed.unsqueeze(1), -1e9)
        a = self.drop(torch.softmax(logits, dim=-1))
        o = (a @ v).transpose(1, 2).reshape(B, N, d)
        H = H + self.drop(self.out(o))
        return H + self.ff(self.norm2(H))


class AttentionNet(nn.Module):
    """Transformer 3D encoder + tabular fusion + regression head.

    Constructor signature is a superset of ``DistanceNet``'s so
    ``train.run_fold`` builds it with the same keyword set; ``n_heads``,
    ``sparse`` and ``bias_max`` are the additions.
    """

    def __init__(self, dim: int = 96, layers: int = 3, dropout: float = 0.1,
                 tabular_dim: int = 0, head_hidden: int = 256,
                 n_z: int = 32, node_feat_dim: int = 5,
                 radial_bins: int = 32, radial_max: float = 8.0,
                 head_embed_mult: int = 1, rbf_bins: int = 32,
                 rbf_max: float = 5.0, pair_head: bool = False,
                 film_dim: int = 0, n_heads: int = 4, sparse: bool = False,
                 bias_max: float = 10.0):
        super().__init__()
        self.dim = dim
        self.sparse = bool(sparse)
        self.cutoff = float(rbf_max)          # the message-passing neighbourhood
        self.z_emb = nn.Embedding(n_z, dim)
        self.node_in = _mlp(node_feat_dim, dim, dim, dropout)
        # edge-distance basis for the readout's edge terms (as in DistanceNet)
        self.rbf = RBFExpansion(rbf_bins, rbf_max)
        # pairwise-distance basis for the attention bias: saturates past
        # bias_max, so a 12 A pair looks like a 10 A pair
        self.pair_rbf = RBFExpansion(rbf_bins, bias_max)
        self.layers = nn.ModuleList(
            [DistanceBiasedAttention(dim, n_heads, rbf_bins, dropout)
             for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)

        # readout: identical to DistanceNet
        self.radial_bins = radial_bins
        centres = torch.linspace(0.0, radial_max, radial_bins)
        self.register_buffer("radial_centres", centres)
        self.radial_width = float(radial_max / max(radial_bins - 1, 1))
        self.radial_proj = _mlp(2 * radial_bins, dim, dim, dropout)
        self.edge_proj = _mlp(rbf_bins, dim, dim, dropout)

        self.embed_dim = 9 * dim
        self.tabular_dim = tabular_dim
        self.head_embed_mult = int(head_embed_mult)
        head_in = self.head_embed_mult * self.embed_dim + tabular_dim
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
        if "coords" not in batch:
            raise ValueError("AttentionNet needs batch['coords']; set "
                             "ComplexCache.with_coords = True")
        hn = self.z_emb(batch["z_idx"]) + self.node_in(batch["node_feat"])
        nb = batch["node_batch"]
        B = int(batch["n_complexes"])
        counts, pos = dense_index(nb, B)
        N = int(counts.max())

        H = hn.new_zeros((B, N, self.dim))
        H[nb, pos] = hn
        present = torch.zeros((B, N), dtype=torch.bool, device=hn.device)
        present[nb, pos] = True
        C = hn.new_zeros((B, N, 3))
        C[nb, pos] = batch["coords"].to(hn.dtype)
        D = torch.cdist(C, C)                                    # (B, N, N)
        allowed = present.unsqueeze(1) & present.unsqueeze(2)    # key & query
        if self.sparse:
            allowed = allowed & (D <= self.cutoff)
        rbf_pair = self.pair_rbf(D.reshape(-1)).view(B, N, N, -1)
        for layer in self.layers:
            H = layer(H, rbf_pair, allowed)
        hn = self.final_norm(H[nb, pos])                         # back to flat

        ei = batch["edge_index"]
        rbf = self.rbf(batch["edge_filt"])
        he = self.edge_proj(rbf)
        eb = batch["edge_batch"]
        parts = [scatter_mean(hn, nb, B), scatter_max(hn, nb, B),
                 scatter_mean(he, eb, B), scatter_max(he, eb, B),
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
        if self.pair_head is None:
            raise ValueError("model built without --pair-head")
        z = torch.cat([emb[i], emb[j], emb[i] - emb[j]], dim=-1)
        return self.pair_head(z).squeeze(-1)

    def modulate(self, e: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.film is None:
            raise ValueError("model built without --film")
        g, b = self.film(cond).chunk(2, dim=-1)
        return (1.0 + g) * e + b
