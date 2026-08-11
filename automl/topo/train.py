#!/usr/bin/env python3
"""Train and evaluate topological models under the existing CV protocol.

The yardstick does not change.  Folds come from
``automl.evaluation.grouped_folds`` (leave-extractants-out), metrics from
``automl.evaluation.full_metrics``, and out-of-fold predictions are written in
the same schema the tabular sweeps use so ``automl.compare.paired_bootstrap``
can pair a topological arm against CatBoost on identical rows.

Efficiency note that is also a correctness note
-----------------------------------------------
4,746 rows are backed by only 953 distinct geometries -- rows sharing a complex
differ only in experimental conditions.  Each complex is therefore encoded
**once** per step and its embedding is gathered for every row that references
it.  That is ~5x less compute, and it makes explicit that the structural sample
size is 953, not 4,746: the number that actually governs overfitting here.

Early stopping uses an inner split that is *also* grouped by extractant, so no
extractant informs the stopping decision for a fold it is evaluated in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from automl import evaluation as ev
from automl.matrix_cache import load_cache
from automl.dataset import BLOCK_PRESETS, GROUP_COL, TARGET
from automl.topo.simplicial_data import (ConformerComplexes,
                                         SimplicialComplexes, collate)
from automl.topo.snn import SimplicialNet, MaskedChargeHead, count_parameters
from automl.topo.pi_cnn import PersistenceImages, PersistenceCNN, PI_PATH
from automl.topo.tabular_net import TabularNet, NullCache
from automl.topo.dist_gnn import DistanceNet

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "automl/artifacts/topo_runs"
PRETRAIN_DIR = REPO / "automl/artifacts/pretrained"
# Fixed, and deliberately independent of the run seed: the pretrained encoder is
# shared across seeds, so tying it to one of them would be misleading about
# which run produced it.
PRETRAIN_SEED = 42


# ---------------------------------------------------------------------------
def geometry_asset(name: str = "full", verbose: bool = False):
    """The simplicial asset for a geometry set, with its OWN cache.

    CAMPAIGN4.  Both ``build_row_table`` (which maps rows to complex indices via
    ``index_of``) and ``ComplexCache`` (which slices the asset by those indices)
    must use the SAME asset, or ``_cplx`` indexes a different complex than it
    names.  That failure is silent -- every row would train on some other
    molecule and the run would look entirely normal.

    The ``cache=`` argument is equally load-bearing: SimplicialComplexes keys
    its triangle-edge cache only by triangle count, so an asset loaded without
    its own path silently reuses the shipped boundary map.
    """
    if name == "full":
        # the published 956-complex asset; the default, so nothing existing moves
        return SimplicialComplexes(verbose=verbose)
    # Every campaign-4 arm -- INCLUDING "shipped" -- is the 627-complex subset.
    # Returning the published asset for "shipped" was a real bug: that arm would
    # have trained on 956 complexes and 4,746 rows while control and neutral
    # trained on 627 and far fewer, so every cross-arm contrast would have
    # compared two different datasets while looking entirely normal.
    from pathlib import Path as _P
    _root2 = _P(__file__).resolve().parents[2]
    if name in ("water", "octanol"):
        # GFN2-xTB re-optimised IN SOLVENT.  Built for the water-octanol
        # reorganisation probe, which tested them as a 22-column TABULAR block
        # and failed (WO_RESULTS.md).  They have never been given to an encoder
        # as a geometry VIEW, which is a different question: the tabular probe
        # asked whether summary statistics of the reorganisation help, this asks
        # whether the solvent-relaxed structure itself is a better input.
        root = _root2 / "automl/artifacts/vr_conformers" / name
        vr = root / "vietoris_rips_inputs.npz"
        if not vr.exists():
            raise SystemExit(f"--geometry {name} needs {vr}")
        return SimplicialComplexes(vr_path=vr,
                                   cache=root / "triangle_edges.npz",
                                   verbose=verbose)
    root = _root2 / "automl/artifacts/vr_neutral" / name
    vr = root / "vietoris_rips_inputs.npz"
    if not vr.exists():
        raise SystemExit(f"--geometry {name} needs {vr}; build it with "
                         f"python3 -m automl.topo.build_vr_neutral --arm {name}")
    return SimplicialComplexes(vr_path=vr, cache=root / "triangle_edges.npz",
                               verbose=verbose)


EDGE_ROOT = REPO / "automl/artifacts/vr_cutoff"


def edge_asset(name: str, verbose: bool = False):
    """A neighbour graph rebuilt past the shipped asset's 4.0 A ceiling.

    Same two load-bearing rules as geometry_asset: build_row_table and
    ComplexCache must see the SAME asset, and the asset must carry its OWN
    triangle-edge cache, because SimplicialComplexes keys that cache by
    triangle count alone and would otherwise reuse the shipped boundary map.

    These assets carry NO triangles by construction (they scale ~r^6), which is
    why main() hard-gates them to --arch dist / --no-triangles.
    """
    # vr_serial arms live in their own root; same zero-triangle format.
    if name in ("serial", "orig"):
        root = REPO / "automl/artifacts/vr_serial" / name
    # vr_gxtb: the same complexes relaxed under g-xTB, and their matched
    # shipped-coordinate control.  Same zero-triangle format again.
    elif name in ("gxtb", "ship", "gxtbh", "shiph", "gxtbs", "ships"):
        # *h arms keep the basin hops; the unsuffixed pair drops them.  Both
        # policies are run because the choice is a real trade-off: dropping
        # hops makes the contrast cleaner (same conformer both sides) but costs
        # ~17 % of the complexes, and at 406 complexes BOTH arms collapse to a
        # negative R2, so complex count is not a free parameter here.
        root = REPO / "automl/artifacts/vr_gxtb" / name
    else:
        root = EDGE_ROOT / name
    vr = root / "vietoris_rips_inputs.npz"
    if not vr.exists():
        raise SystemExit(f"--edge-asset {name} needs {vr}; build it with "
                         f"python3 -m automl.topo.build_neighbor_graph "
                         f"--name {name} --cutoff ...")
    return SimplicialComplexes(vr_path=vr, cache=root / "triangle_edges.npz",
                               verbose=verbose)


def build_row_table(preset: str = "baseline_2d", arch: str = "snn",
                    match_rows: str = "snn", geometry: str = "full",
                    edge_asset_name: str | None = None
                    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Rows backed by this arm's 3D asset, plus their tabular design matrix.

    The two arms key off different assets (956 VR complexes vs 953 persistence
    images), so the eligible row set differs slightly.  Both are joined on
    ``geometry_feature_build_id``, which reaches 4,746 rows -- ``build_id``
    only reaches 4,402 and would silently drop 344 rows.

    ``arch="tabular"`` uses no 3D asset at all, but it still has to be scored on
    the *same rows* as the arm it is a control for, or the paired bootstrap
    silently compares two different datasets (4,746 vs 4,742 rows here).  So the
    asset is still consulted, purely to select rows -- ``match_rows`` names
    which arm's eligibility to reproduce.
    """
    df, blocks, _ = load_cache()
    # 'dist' reads the same Vietoris-Rips asset as 'snn' -- same edges, same
    # node features -- so it must select the same rows or the paired bootstrap
    # would compare two different datasets.
    key_arch = match_rows if arch == "tabular" else (
        "snn" if arch == "dist" else arch)
    asset = ((edge_asset(edge_asset_name) if edge_asset_name
              else geometry_asset(geometry)) if key_arch == "snn"
             else PersistenceImages())
    key = df["geometry_feature_build_id"].astype(str)
    df = df.assign(_cplx=[asset.index_of(k) for k in key])
    df = df[df["_cplx"].notna() & df["geometry_ok"].astype(bool)].reset_index(drop=True)
    df["_cplx"] = df["_cplx"].astype(int)
    cols = blocks.select(BLOCK_PRESETS[preset])
    X = df[cols].to_numpy(dtype=np.float32)
    # median-impute + standardise; statistics are refit per fold in run_fold to
    # avoid leaking test-fold statistics into training.
    return df, X, cols


def aux_target_columns(df: pd.DataFrame, name: str) -> np.ndarray:
    """Per-row auxiliary target, standardised, with NaN where unavailable.

    SWEEP2 axis B.  No multi-task setup has ever been run in this study: xTB
    charge, E_int, coordination number and CShM are all *inputs* somewhere and
    none has ever been a training target.  That matters most for the energies --
    as inputs they destroyed the adjacent-pair metric by substituting for the
    exact ionic radius (ENERGY_RESULTS.md), but as targets they cannot enter the
    prediction path at all and can only shape the encoder.

    Rows whose target is missing are returned as NaN and masked out of the loss
    rather than imputed, because imputing a physical quantity to its mean would
    teach the encoder that every such complex is average.
    """
    if name == "cshm":
        # Continuous shape measures are defined per coordination number, so most
        # reference columns are NaN for any one complex.  The chemically
        # meaningful scalar is the distance to the NEAREST ideal polyhedron.
        cols = [c for c in df.columns if c.startswith("g3__polyhedron__cshm")]
        if not cols:
            raise SystemExit("no g3 CShM columns; rebuild the matrix cache")
        y = df[cols].min(axis=1, skipna=True).to_numpy(dtype=np.float64)
    elif name == "eint":
        cols = ["gE__abs__e_int_water_ev", "gE__abs__dg_transfer_ev"]
    elif name == "qtransfer":
        cols = ["gE__abs__q_metal_water", "gE__abs__q_transfer_water"]
    else:
        raise ValueError(name)
    if name != "cshm":
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"aux target {name!r} needs {missing}; run "
                             f"automl.qc.energy_features and rebuild the cache")
        y = df[cols].to_numpy(dtype=np.float64)
    y = y.reshape(len(df), -1)
    mu = np.nanmean(y, axis=0)
    sd = np.nanstd(y, axis=0)
    sd[sd < 1e-8] = 1.0
    return ((y - mu) / sd).astype(np.float32)


def block_means(pred: torch.Tensor, tgt: torch.Tensor, bidx: torch.Tensor,
                n_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block means of the prediction and the target.

    The level term of the decomposed objective.  Returns one value per block,
    not one per row: a composition block is a single nuisance parameter however
    many measurements happen to sit in it, and weighting it by row count would
    hand the largest blocks the same dominance the plain MSE already gives them.

    Kept at module level so the arithmetic can be tested without standing up a
    fold, a cache and a GPU.
    """
    cnt = torch.zeros(n_blocks, device=pred.device, dtype=pred.dtype).index_add_(
        0, bidx, torch.ones_like(pred)).clamp(min=1.0)
    pm = torch.zeros(n_blocks, device=pred.device,
                     dtype=pred.dtype).index_add_(0, bidx, pred) / cnt
    tm = torch.zeros(n_blocks, device=tgt.device,
                     dtype=tgt.dtype).index_add_(0, bidx, tgt) / cnt
    return pm, tm


def _standardise(train_X, *others):
    if train_X.shape[1] == 0:                      # topology-only ablation
        return [train_X] + [o for o in others]
    med = np.nanmedian(train_X, axis=0)
    # A column that is all-NaN in the TRAINING FOLD has no median, and NaN
    # imputed there propagates through the head into every prediction -- the
    # whole run returns NaN.  It bit the shape preset immediately: four CShM
    # reference columns (COC, OC, PBPY, TPR) are all-NaN across the geometry-OK
    # rows because continuous shape measures are defined per coordination number
    # and no modelled complex has those. build_matrix's own all-NaN filter runs
    # on the full 5,992-row table and cannot see it.
    #
    # Imputing 0 makes such a column constant, which the sd guard below then
    # turns into a no-op contribution -- the honest behaviour for a feature that
    # carries nothing.  baseline_2d never hit this because ECFP, RDKit and cond
    # are dense.
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(train_X), train_X, med)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    out = [(Xtr - mu) / sd]
    for o in others:
        o2 = np.where(np.isfinite(o), o, med)
        out.append((o2 - mu) / sd)
    return out


# ---------------------------------------------------------------------------
class ImageCache:
    """Batches persistence images; same interface as ComplexCache."""

    def __init__(self, P: PersistenceImages, device):
        self.P, self.device = P, device

    def batch(self, ids: list[int], conformers: list[int] | None = None):
        # Signature matches ComplexCache.  Persistence images are one per
        # complex, so there is nothing to select a conformer from; the argument
        # is accepted and ignored rather than rejected, because the training
        # loop passes it unconditionally.
        #
        # This was a live regression: `--conformers` added a second positional
        # argument at the call site without widening ImageCache or NullCache, so
        # `--arch picnn` and `--arch tabular` both raised TypeError from 33324ea
        # onwards.  The published P0 and T0w runs predate that commit, which is
        # why it went unnoticed -- nothing re-ran those arms until now.
        if conformers is not None and any(c != 0 for c in conformers):
            raise ValueError("persistence images have no conformer axis")
        return self.P.batch(ids, self.device)


class ComplexCache:
    """Pre-loads and caches collated complexes on the target device."""

    def __init__(self, S: SimplicialComplexes, filtration_max, heavy_only,
                 device, angular: bool = False,
                 node_angular: bool | None = None,
                 metal_angular: bool | None = None):
        self.S, self.device = S, device
        self.filtration_max, self.heavy_only = filtration_max, heavy_only
        # SWEEP2 axis A: build the angular features once per complex here, where
        # the cache pays for them once instead of once per batch.  The two
        # blocks are separate switches because they are separate experiments:
        # node_ang widens node_feat and so changes the node encoder, whereas
        # metal_ang only adds a readout key and leaves node_feat alone.
        self.node_angular = bool(angular if node_angular is None else node_angular)
        self.metal_angular = bool(angular if metal_angular is None else metal_angular)
        self._c: dict[Any, Any] = {}

    def get(self, k: int, conformer: int = 0):
        key = (k, conformer)
        if key not in self._c:
            if conformer and hasattr(self.S, "n_conformers"):
                self._c[key] = self.S.get(k, conformer=conformer,
                                          filtration_max=self.filtration_max,
                                          heavy_only=self.heavy_only)
            else:
                self._c[key] = self.S.get(k, filtration_max=self.filtration_max,
                                          heavy_only=self.heavy_only,
                                          node_angular=self.node_angular,
                                          metal_angular=self.metal_angular)
        return self._c[key]

    def n_conformers(self, k: int) -> int:
        return self.S.n_conformers(k) if hasattr(self.S, "n_conformers") else 1

    def batch(self, ids: list[int], conformers: list[int] | None = None):
        cs = conformers if conformers is not None else [0] * len(ids)
        b = collate([self.get(i, c) for i, c in zip(ids, cs)])
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in b.items()}


# ---------------------------------------------------------------------------
def pretrain(model: SimplicialNet, cache: ComplexCache, complex_ids: list[int],
             *, epochs: int, batch_size: int, lr: float, device, seed: int,
             log_every: int = 5) -> None:
    """Self-supervised warm-up: reconstruct masked charges and edge radii.

    953 structures is thin for an end-to-end encoder.  These targets are free,
    defined on every complex including ones whose rows are in no training fold,
    and force the encoder to represent local chemical environment before it
    ever sees a log D value.  No target information is involved, so this is
    run once outside the CV loop.
    """
    torch.manual_seed(seed)
    head = MaskedChargeHead(model.dim).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        order = rng.permutation(complex_ids)
        tot, nb = 0.0, 0
        for s in range(0, len(order), batch_size):
            ids = [int(i) for i in order[s:s + batch_size]]
            # Pretraining benefits most from the conformers: its targets are
            # free and defined on every structure, so the extra geometries are
            # ~2,800 training examples rather than ~950, with no log D involved.
            confs = [int(rng.integers(0, cache.n_conformers(i))) for i in ids]
            b = cache.batch(ids, confs)
            q_true = b["node_feat"][:, 0].clone()
            f_true = b["edge_filt"].squeeze(-1).clone()
            # Never score reconstruction on an imputed charge: the target is a
            # placeholder, not a measurement, and asking the encoder to predict
            # it teaches it to reproduce the imputation.
            observed = b["node_feat"][:, 1] < 0.5
            mask = (torch.rand_like(q_true) < 0.25) & observed
            nf = b["node_feat"].clone()
            nf[mask, 0] = 0.0
            if not bool(mask.any()):
                continue
            b2 = dict(b); b2["node_feat"] = nf
            hn = model.z_emb(b2["z_idx"]) + model.node_in(b2["node_feat"])
            he = model.edge_in(b2["edge_filt"])
            ht = model.tri_in(b2["tri_filt"])
            for layer in model.layers:
                hn, he, ht = layer(hn, he, ht, b2["edge_index"], b2["tri_edges"])
            q_hat, f_hat = head(hn, he)
            loss = (nn.functional.mse_loss(q_hat[mask], q_true[mask])
                    + 0.1 * nn.functional.mse_loss(f_hat, f_true))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"    [pretrain] epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}",
                  flush=True)


# ---------------------------------------------------------------------------
def rebuild_from_differences(means: dict[int, float],
                             diffs: dict[tuple[int, int], float]
                             ) -> dict[int, float]:
    """Replace a chain's adjacent increments while preserving its overall level.

    ``means`` maps metal index -> the level head's cell mean.  ``diffs`` maps
    (lighter, heavier) -> the predicted ``y_lighter - y_heavier``.  Walking the
    chain reproduces every supplied difference exactly; gaps with no supplied
    difference keep the level head's own spacing.  The result is then shifted so
    its mean equals the input's, because the metric scores DIFFERENCES and the
    block's absolute level is the level head's job, not the pair head's.

    Extracted from ``run_fold`` so the arithmetic can be tested without a GPU.
    """
    ks = sorted(means)
    new = dict(means)
    for a_, b_ in zip(ks[:-1], ks[1:]):
        if (a_, b_) in diffs:
            new[b_] = new[a_] - float(diffs[(a_, b_)])
        else:
            new[b_] = new[a_] + (means[b_] - means[a_])
    off = (np.mean([means[k] for k in ks]) - np.mean([new[k] for k in ks]))
    return {k: new[k] + off for k in ks}


def _cond_columns(cfg) -> list[int]:
    """Design-matrix positions of the experimental-condition columns (T3).

    Resolved by NAME, so a preset change cannot silently point FiLM at the
    wrong block.
    """
    return [i for i, c in enumerate(cfg.get("_cols") or [])
            if str(c).startswith("cond__")]


# The 'metal' block, verbatim from dataset.py.  Resolved by NAME like the
# conditions, so a preset change cannot silently point the interaction head at
# the wrong columns -- and note dataset.py drops constant columns, so metal_ox
# is legitimately absent (every row is Ln(III)).
METAL_COLS = ("Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal",
              "metal_ox")
RADIUS_COL = "Ionic Radius_metal"
LIDX_COL = "lanthanide_index"


def _metal_columns(cfg) -> list[int]:
    """Every column whose value is a property of the METAL, not the ligand.

    The mphys__ block belongs here and its omission was a real defect, found by
    measuring how block-constant u actually was rather than assuming it: those
    columns are per-lanthanide lookups, so leaving them in u made u vary within
    a block and broke the identity the interaction head exists to enforce.
    They are also precisely the material the phi basis is built from, so
    leaving them in u would let g read its own argument.
    """
    return [i for i, c in enumerate(cfg.get("_cols") or [])
            if c in METAL_COLS or str(c).startswith("mphys__")]


def _named_column(cfg, name: str) -> int:
    cols = list(cfg.get("_cols") or [])
    if name not in cols:
        raise SystemExit(
            f"--radius-slope needs the column {name!r}, which is not in this "
            f"preset. It lives in the 'metal' block; --topology-only and any "
            f"preset without 'metal' cannot use the interaction head.")
    return cols.index(name)


def run_fold(df, X, cache, tr_idx, te_idx, *, cfg, device, seed,
             pretrained_state=None, target_col: str | None = None,
             emb_out: np.ndarray | None = None) -> np.ndarray:
    """Train on tr_idx, predict te_idx.

    ``emb_out``, when given, receives this fold's **test-row** encoder
    embeddings.  They come from a model that never saw those extractants, so the
    assembled out-of-fold embedding matrix is leakage-free in the same sense the
    out-of-fold predictions are -- which is what makes it legitimate to hand to a
    downstream learner.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # inner validation split, grouped by extractant -- never by row
    if cfg.get("no_inner_val"):
        # Capacity check only.  Holding out 15% of groups and then scoring all
        # of tr_idx means ~15% of the scored rows were never trained on, which
        # caps train-on-train R2 near 0.89 no matter how well the model fits --
        # exactly the plateau the epoch sweep showed.
        fit_idx = val_idx = tr_idx
    else:
        g_tr = df[GROUP_COL].to_numpy()[tr_idx]
        uniq = np.unique(g_tr)
        val_groups = set(rng.choice(uniq, size=max(1, int(0.15 * len(uniq))),
                                    replace=False).tolist())
        is_val = np.array([g in val_groups for g in g_tr])
        fit_idx, val_idx = tr_idx[~is_val], tr_idx[is_val]

    Xtr, Xval, Xte = _standardise(X[fit_idx], X[val_idx], X[te_idx])
    # Before any architecture is built: both the training loop and _predict use
    # it for every arch, so defining it inside the snn branch would raise
    # NameError only on --arch dist/tabular.
    cond_idx = _cond_columns(cfg)
    if cfg.get("film") and not cond_idx:
        raise SystemExit("--film needs cond__ columns; none in this preset")
    y = df[target_col or TARGET].to_numpy(dtype=np.float32)
    ymu, ysd = float(y[fit_idx].mean()), float(y[fit_idx].std() or 1.0)

    if cfg.get("arch", "snn") == "tabular":
        model = TabularNet(dim=cfg["dim"], dropout=cfg["dropout"],
                           tabular_dim=X.shape[1],
                           head_hidden=cfg["head_hidden"]).to(device)
    elif cfg.get("arch", "snn") == "picnn":
        model = PersistenceCNN(dim=cfg["dim"], dropout=cfg["dropout"],
                               tabular_dim=X.shape[1],
                               head_hidden=cfg["head_hidden"],
                               in_channels=cfg.get("in_channels", 1)).to(device)
    elif cfg.get("arch", "snn") == "dist":
        # Same edges, same node inputs, same readout, same head width -- the
        # only difference from the SNN is that there is no simplicial structure.
        # That is what makes it a control for "is it simplicial, or just 3D?".
        model = DistanceNet(dim=cfg["dim"], layers=cfg["layers"],
                            dropout=cfg["dropout"], tabular_dim=X.shape[1],
                            head_hidden=cfg["head_hidden"],
                            rbf_bins=int(cfg.get("rbf_bins") or 32),
                            # Unset = --filtration-max, which is what every
                            # published dist run used.  It matters once the
                            # graph widens: edges past rbf_max all land in the
                            # saturated tail of the basis and the filter net
                            # cannot tell 6 A from 9 A.
                            rbf_max=float(cfg["rbf_max"]
                                          if cfg.get("rbf_max") is not None
                                          else cfg.get("filtration_max", 3.5)),
                            # SWEEP2's C1 cell was SNN-only, so dist has carried
                            # DistanceNet's own 32 / 8.0 in every run.  Passing
                            # them through makes the axis testable on the
                            # encoder that has no triangles to confound it.
                            radial_bins=int(cfg.get("radial_bins") or 32),
                            radial_max=float(cfg.get("radial_max") or 8.0),
                            head_embed_mult=2 if cfg.get("block_centre") else 1,
                            # Was omitted, silently disabling --pair-head on
                            # every --arch dist run while still recording
                            # pair_head=True in the config.
                            pair_head=bool(cfg.get("pair_head")),
                            ).to(device)
    else:
        from automl.topo.simplicial_data import N_ANGULAR_BINS
        # T3: which design-matrix columns are experimental conditions.  Taken
        # from the standardised matrix so FiLM sees the same scaling the head
        # does, and resolved by name so a preset change cannot silently shift
        # them.
        model = SimplicialNet(dim=cfg["dim"], layers=cfg["layers"],
                              dropout=cfg["dropout"], tabular_dim=X.shape[1],
                              head_hidden=cfg["head_hidden"],
                              head_embed_mult=2 if cfg.get("block_centre") else 1,
                              use_triangles=not cfg.get("no_triangles", False),
                              node_feat_dim=5 + (N_ANGULAR_BINS
                                                 if cfg.get("node_angular") else 0),
                              pair_head=bool(cfg.get("pair_head")),
                              film_dim=(len(cond_idx) if cfg.get("film") else 0),
                              angular_readout=bool(cfg.get("angular_readout")),
                              attn_pool=bool(cfg.get("attn_pool")),
                              angular_bins=N_ANGULAR_BINS,
                              radial_bins=int(cfg.get("radial_bins") or 32),
                              radial_max=float(cfg.get("radial_max") or 8.0),
                              ).to(device)
    if pretrained_state is not None:
        # Encoder weights only.  Pretraining has no tabular block, so its head
        # is sized for embed_dim alone (768) while the fold model's head takes
        # embed_dim + 746 tabular features (1514).  strict=False does not
        # forgive a *shape* mismatch, only a missing key -- so the head must be
        # filtered out explicitly rather than left to fail at load time.
        enc = {k: v for k, v in pretrained_state.items()
               if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(enc, strict=False)
        bad = [k for k in unexpected if not k.startswith("head.")]
        if bad:
            raise RuntimeError(f"pretrained weights did not map onto the "
                               f"encoder: unexpected {bad[:5]}")
    # SWEEP2 axis B: a second head on the encoder embedding, trained jointly.
    aux_name = cfg.get("aux_target")
    aux_w = float(cfg.get("aux_weight") or 0.0)
    aux_head = aux_y = None
    if aux_name:
        aux_y = aux_target_columns(df, aux_name)          # (rows, k), NaN allowed
        # ``_centre`` concatenates the block-centred deviation when
        # --block-centre is on, doubling the width the head sees.  Size from the
        # multiplier rather than from embed_dim, or the two would silently
        # mismatch in exactly that combination.
        aux_in = model.embed_dim * getattr(model, "head_embed_mult", 1)
        aux_head = nn.Linear(aux_in, aux_y.shape[1]).to(device)

    # --- CAMPAIGN6: the radius-interaction head ------------------------------
    #
    #     pred = f(u) + sum_k g_k(u) * phi_k(metal)
    #
    # where u is the row representation made METAL-FREE and phi is a small basis
    # of clean per-metal scalars.  Within a composition block only phi varies,
    # so the predicted adjacent difference is exactly sum_k g_k(u) * d phi_k --
    # one ligand-level selectivity coefficient times a known series step.
    #
    # Why this is not --pair-head (T2), which lost at -0.0253/-0.0321/-0.0832:
    # T2 put 254k parameters on a pathway evaluation never reads, which is why
    # --pair-reconcile had to be invented and why it then failed at -1.30
    # ("there is no skill to route").  This head lives INSIDE the level
    # prediction.  f and g are both trained on every row through the level task,
    # and the metric reads f + sum g*phi, the same scalar the level head emits.
    # It CONSTRAINS within-block variation rather than adding a pathway, and
    # unconstrained within-block variation is the failure mode measured at
    # -0.3167 (the A1 collapse) and attributed at 93%.
    #
    # Why rank K and not a single slope: mean dy by pair index is non-monotone
    # and changes sign across the series, so g*dr with a near-constant dr cannot
    # express its shape.  Physically the right form too -- lanthanide strain is
    # a cavity-mismatch energy ~ k(r-r0)^2, whose derivative 2k(r-r0) IS a
    # ligand-dependent selectivity slope carrying both a stiffness and a
    # preferred radius.
    slope_kind = cfg.get("radius_slope") or "off"
    slope_u_mode = cfg.get("radius_slope_u") or "block"
    slope_heads: list[nn.Module] = []
    slope_basis = None
    struct_w = model.embed_dim * getattr(model, "head_embed_mult", 1)
    u_mask = None
    if slope_kind != "off":
        if X.shape[1] == 0:
            raise SystemExit("--radius-slope needs the tabular block; it "
                             "cannot be combined with --topology-only")
        r_col = _named_column(cfg, RADIUS_COL)
        l_col = _named_column(cfg, LIDX_COL)
        head_in = struct_w + X.shape[1]

        def _slope_mlp() -> nn.Module:
            return nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, cfg["head_hidden"]), nn.SiLU(),
                nn.Dropout(cfg["dropout"]),
                nn.Linear(cfg["head_hidden"], cfg["head_hidden"] // 2),
                nn.SiLU(),
                nn.Linear(cfg["head_hidden"] // 2, 1)).to(device)

        # phi, built from columns that are already standardised per fold.  The
        # quadratic term is the cavity-mismatch one; the |lanthanide_index|
        # terms give the basis a coordinate that is NOT collinear with radius,
        # which is what lets it bend where the series does.
        if slope_kind == "linear":
            slope_basis = [("r", lambda er: er[:, struct_w + r_col])]
        elif slope_kind == "quad":
            slope_basis = [("r", lambda er: er[:, struct_w + r_col]),
                           ("r2", lambda er: er[:, struct_w + r_col] ** 2)]
        else:                                              # "basis"
            slope_basis = [
                ("r", lambda er: er[:, struct_w + r_col]),
                ("r2", lambda er: er[:, struct_w + r_col] ** 2),
                ("l", lambda er: er[:, struct_w + l_col]),
                ("l2", lambda er: er[:, struct_w + l_col] ** 2),
            ]
        slope_heads = [_slope_mlp() for _ in slope_basis]
        # u is metal-free.  Zeroing the metal COLUMNS is necessary but not
        # sufficient: each (ligand, metal) is a distinct complex and the encoder
        # embeds Z, so the structural half is metal-dependent too.  That half is
        # therefore replaced by its composition-block mean in _slope_terms --
        # without which the "difference is exactly sum g*dphi" identity, the
        # only reason to build this head, quietly does not hold.
        u_mask = torch.ones(head_in, device=device)
        for c in _metal_columns(cfg):
            u_mask[struct_w + c] = 0.0
        print(f"    [radius-slope {slope_kind}/{slope_u_mode}] "
              f"K={len(slope_basis)} "
              f"({', '.join(n for n, _ in slope_basis)}), "
              f"{len(_metal_columns(cfg))} metal columns masked", flush=True)

    opt = torch.optim.AdamW(list(model.parameters())
                            + (list(aux_head.parameters()) if aux_head else [])
                            + [p for h in slope_heads for p in h.parameters()],
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    cplx = df["_cplx"].to_numpy()
    best_val, best_state, patience = float("inf"), None, 0

    # ---- composition blocks, for the pairwise-contrast loss -----------------
    # The adjacent-pair metric scores predicted *differences* in log D between
    # two lanthanides sharing an extractant and conditions.  Nothing in the
    # plain regression objective ever looks at that quantity: a model can fit
    # absolute log D well while its within-block contrasts are noise.  Batching
    # by composition block and adding a loss on the differences trains the
    # thing that is actually measured.
    #
    # Legality: this uses only training-fold rows, and the blocks come from
    # composition_key, which is experimental metadata, not the target.
    comp_all = df["composition_key"].to_numpy()
    lidx_all = df["lanthanide_index"].to_numpy()
    # Which blocking to train and early-stop against.  ``composition_key`` bins
    # the conditions; ``strict_composition_key`` does not, and dataset.py:387
    # argues the binned one "turns a real log D difference into label noise on a
    # zero-difference feature vector".  Default is the published key, so nothing
    # existing moves.
    blk_all = (df[cfg["block_key"]].to_numpy()
               if cfg.get("block_key") else comp_all)
    pair_w = float(cfg.get("pair_loss_weight", 0.0))
    ph_w = float(cfg.get("pair_head_weight") or 0.0)
    # CAMPAIGN6 contrast-term shape.  Every default reproduces the published
    # term exactly -- 3.0 was hardcoded, "sq" was the only error, and no run to
    # date collapsed replicates.  See the loss block for the census that
    # motivates --pair-metric-align.
    pair_adj_w = float(cfg.get("pair_adj_weight", 3.0))
    pair_adj_only = bool(cfg.get("pair_adj_only"))
    pair_kind = cfg.get("pair_loss_kind") or "sq"
    pair_delta = float(cfg.get("pair_huber_delta") or 1.0)
    pair_align = bool(cfg.get("pair_metric_align"))
    pair_subsample = float(cfg.get("pair_subsample") or 1.0)
    level_loss = cfg.get("level_loss") or "huber"
    level_delta = float(cfg.get("level_huber_delta") or 1.0)
    level_q = float(cfg.get("level_quantile") or 0.5)
    # None = the published objective (Huber on the raw target).  A number turns
    # on the decomposed objective and is the weight on the block-mean term.
    level_w = cfg.get("level_weight")
    level_w = None if level_w is None else float(level_w)
    fit_blocks: list[np.ndarray] = []
    singleton_rows = np.array([], dtype=int)
    if pair_w > 0:
        pos = {int(r): i for i, r in enumerate(fit_idx)}
        by_block: dict[Any, list[int]] = {}
        for r in fit_idx:
            by_block.setdefault(blk_all[r], []).append(int(r))
        fit_blocks = [np.array(v) for v in by_block.values() if len(v) >= 2]
        # Only the decomposed objective has a term these rows can contribute to.
        singleton_rows = (np.array([v[0] for v in by_block.values()
                                    if len(v) == 1], dtype=int)
                          if cfg.get("level_weight") is not None
                          else np.array([], dtype=int))

    # --- conformer ensembling and block-centred embeddings -------------------
    n_conf = int(cfg.get("n_conformers", 1))
    block_centre = bool(cfg.get("block_centre", False))

    def _encode(ids: list[int], conformers: list[int] | None = None):
        """Complex embeddings, averaged over conformers when asked.

        At inference every available conformer is encoded and mean-pooled: the
        re-optimised geometries differ from the shipped one by ~0.3 A in mean
        M-L distance while the adjacent-lanthanide signal is ~0.013 A, so a
        single conformer is a high-variance estimate of the quantity the model
        needs.  Averaging is the cheapest available variance reduction and it is
        what makes the extra structures worth having.
        """
        if conformers is not None:
            return model.encode(cache.batch(ids, conformers))
        if n_conf <= 1:
            return model.encode(cache.batch(ids))
        acc, cnt = None, 0
        for c in range(n_conf):
            # Complexes with fewer conformers wrap back to the shipped geometry
            # rather than being skipped, so every complex keeps equal weight.
            e = model.encode(cache.batch(ids, [c] * len(ids)))
            acc = e if acc is None else acc + e
            cnt += 1
        return acc / max(cnt, 1)

    def _centre(emb_rows: torch.Tensor, blocks: np.ndarray) -> torch.Tensor:
        """Concatenate each row's embedding with its deviation from its block.

        The metric scores *differences* between two lanthanides sharing an
        extractant and conditions, so the ligand and much of the conformer noise
        is common-mode within a composition block and cancels in the deviation.
        Concatenated rather than substituted because 347 of 552 blocks hold a
        single metal, where the deviation is identically zero -- replacing would
        throw their absolute embedding away.

        Legal under the split: composition keys nest strictly inside extractants
        (552 blocks, none spanning two), so a block never straddles a fold and
        the mean is always taken over rows from the same fold.
        """
        if not block_centre:
            return emb_rows
        codes, _ = pd.factorize(blocks)
        idx = torch.as_tensor(codes, device=emb_rows.device, dtype=torch.long)
        n = int(codes.max()) + 1 if len(codes) else 0
        sums = torch.zeros((n, emb_rows.shape[1]), device=emb_rows.device,
                           dtype=emb_rows.dtype).index_add_(0, idx, emb_rows)
        counts = torch.zeros(n, device=emb_rows.device, dtype=emb_rows.dtype
                             ).index_add_(0, idx, torch.ones_like(idx,
                                                                 dtype=emb_rows.dtype))
        means = sums / counts.clamp(min=1.0).unsqueeze(-1)
        return torch.cat([emb_rows, emb_rows - means[idx]], dim=-1)

    def _slope_terms(er: torch.Tensor, blocks: np.ndarray):
        """sum_k g_k(u) * phi_k for one batch of rows, or None if off."""
        if not slope_heads:
            return None
        u = er * u_mask
        if slope_u_mode == "block":
            # Make the STRUCTURAL half metal-free by replacing it with its
            # composition-block mean.  Zeroing the metal columns handles the
            # tabular half only; the embedding is a different complex per metal.
            # Exact here because every call site passes whole blocks.
            codes, _ = pd.factorize(blocks)
            bi = torch.as_tensor(codes, device=u.device, dtype=torch.long)
            n = int(codes.max()) + 1 if len(codes) else 0
            sums = torch.zeros((n, struct_w), device=u.device, dtype=u.dtype
                               ).index_add_(0, bi, u[:, :struct_w])
            cnt = torch.zeros(n, device=u.device, dtype=u.dtype).index_add_(
                0, bi, torch.ones_like(bi, dtype=u.dtype)).clamp(min=1.0)
            u = torch.cat([(sums / cnt.unsqueeze(-1))[bi], u[:, struct_w:]],
                          dim=-1)
        out = None
        for h, (_name, phi) in zip(slope_heads, slope_basis):
            term = h(u).squeeze(-1) * phi(er)
            out = term if out is None else out + term
        return out

    def _slope_mode(train: bool) -> None:
        """The slope MLPs carry Dropout; left in train mode every prediction
        would be stochastic. aux_head is a bare Linear, which is why it never
        needed this."""
        for h in slope_heads:
            h.train() if train else h.eval()

    def _eval_chunks(idx) -> list[np.ndarray]:
        """Positions in ``idx``, grouped so no composition block is ever split.

        Blocks are *packed* into shared batches rather than sent one at a time.
        Both are exact -- ``_centre`` factorises within the batch, so several
        whole blocks coexist safely -- but one-block-at-a-time costs 270
        ``encode`` calls per fold against 12 packed, and each call is tiny
        enough that the GPU is idle between them.

        Splitting a block across batches would not be exact: the block mean
        would then depend on the batch size, which is not a property of the
        data, and training (which always sees whole blocks) and inference would
        compute different features from the same rows.
        """
        # The interaction head's block-mean u has the same requirement as
        # --block-centre: a block split across chunks would give a
        # batch-size-dependent mean, so training and inference would compute
        # different features from the same rows.
        if not (block_centre or (slope_heads and slope_u_mode == "block")):
            step = max(cfg["eval_batch"], 1)
            return [np.arange(s, min(s + step, len(idx)))
                    for s in range(0, len(idx), step)]
        budget = max(cfg["eval_batch"], 1)
        chunks, buf = [], []
        for _key, positions in pd.Series(blk_all[idx]).groupby(
                blk_all[idx], sort=False).groups.items():
            pos = np.asarray(positions, dtype=int)
            if buf and len(buf) + len(pos) > budget:
                chunks.append(np.concatenate(buf)); buf = []
            buf.append(pos)
        if buf:
            chunks.append(np.concatenate(buf))
        return chunks

    def _predict(idx, Xs):
        model.eval()
        _slope_mode(False)
        outs = np.empty(len(idx), dtype=np.float64)
        with torch.no_grad():
            for take in _eval_chunks(idx):
                rows = idx[take]
                ids = sorted(set(cplx[rows].tolist()))
                remap = {c: i for i, c in enumerate(ids)}
                emb = _encode(ids)
                gather = torch.as_tensor([remap[c] for c in cplx[rows]],
                                         device=device)
                e = _centre(emb[gather], blk_all[rows])
                tab = torch.as_tensor(Xs[take], device=device)
                if cond_idx and getattr(model, "film", None) is not None:
                    e = model.modulate(e, tab[:, cond_idx])
                er = torch.cat([e, tab], -1)
                p = model.head(er).squeeze(-1)
                st = _slope_terms(er, blk_all[rows])
                if st is not None:
                    p = p + st
                outs[take] = p.cpu().numpy()
        return outs * ysd + ymu

    def _reconcile(level_pred, idx, Xs):
        """Make the pair head's skill reach the metric.

        The metric averages predictions per (block, metal) and differences
        neighbours.  A pair head trained to predict dy sits on a different
        pathway entirely, so however well it does, the scored quantity never
        sees it -- which is the most likely reason T2/T2W/T2X all lost.

        Fix: keep the level head's overall level inside each block, and replace
        its ADJACENT INCREMENTS by the pair head's.  Exact, not a fit: walking
        the chain from the block mean reproduces the requested differences by
        construction, and non-adjacent gaps keep the level head's own spacing.
        Rows inherit their cell's shift, so per-row predictions -- what the
        metric consumes -- carry it.
        """
        model.eval()
        blk = blk_all[idx]
        lidx = lidx_all[idx]
        adj = np.array(level_pred, dtype=np.float64, copy=True)
        with torch.no_grad():
            for b in np.unique(blk):
                sel = np.flatnonzero(blk == b)
                metals = np.unique(lidx[sel])
                if len(metals) < 2:
                    continue
                # cell means from the level head
                m = {int(k): float(adj[sel][lidx[sel] == k].mean())
                     for k in metals}
                ks = sorted(m)
                # one representative row per metal, for the pair head
                rep = {int(k): sel[np.flatnonzero(lidx[sel] == k)[0]]
                       for k in metals}
                ids = sorted({int(cplx[idx[r]]) for r in rep.values()})
                remap = {c: i for i, c in enumerate(ids)}
                emb = _encode(ids)
                rows_r = np.array([idx[rep[k]] for k in ks])
                gather = torch.as_tensor([remap[int(cplx[r])] for r in rows_r],
                                         device=device)
                e = _centre(emb[gather], blk_all[rows_r])
                pos = np.array([int(np.flatnonzero(idx == r)[0])
                                for r in rows_r])
                tab = torch.as_tensor(Xs[pos], device=device)
                if cond_idx and getattr(model, "film", None) is not None:
                    e = model.modulate(e, tab[:, cond_idx])
                er = torch.cat([e, tab], -1)
                diffs = {}
                for a_, b_ in zip(range(len(ks) - 1), range(1, len(ks))):
                    if ks[b_] - ks[a_] != 1:
                        continue          # non-adjacent: keep the level spacing
                    diffs[(ks[a_], ks[b_])] = float(model.pair_forward(
                        er, torch.tensor([a_], device=device),
                        torch.tensor([b_], device=device)).item()) * ysd
                new = rebuild_from_differences(m, diffs)
                for k in ks:
                    adj[sel[lidx[sel] == k]] += new[k] - m[k]
        return adj

    # Row index -> position in the standardised training matrix.
    fit_pos = {int(r): i for i, r in enumerate(fit_idx)}

    def _batches(epoch_rng):
        """Yield row-index arrays for one epoch."""
        if pair_w <= 0 or not fit_blocks:
            order = epoch_rng.permutation(len(fit_idx))
            for s in range(0, len(order), cfg["batch_rows"]):
                yield fit_idx[order[s:s + cfg["batch_rows"]]]
            return
        # Whole composition blocks per batch, so same-block pairs always exist.
        border = epoch_rng.permutation(len(fit_blocks))
        buf: list[int] = []
        for bi in border:
            buf.extend(fit_blocks[bi].tolist())
            if len(buf) >= cfg["batch_rows"]:
                yield np.array(buf); buf = []
        if len(buf) >= 2:
            yield np.array(buf)
        if singleton_rows.size:
            # Rows in single-member blocks contribute no pair, so the published
            # contrast objective never batches them -- it has no term they could
            # enter.  The DECOMPOSED objective does: they carry the block-mean
            # (level) term.  Dropping them there would confound the blocking
            # with the training-set size, which matters because the strict key
            # has 1,573 singleton blocks against the binned key's 202 -- 67% of
            # rows usable versus 96%.  A strict-vs-binned comparison that also
            # varied the row count by a third would be measuring two things.
            #
            # Gated on level_w so the published path is byte-unchanged.
            order = epoch_rng.permutation(len(singleton_rows))
            for s0 in range(0, len(order), cfg["batch_rows"]):
                yield singleton_rows[order[s0:s0 + cfg["batch_rows"]]]

    for ep in range(cfg["epochs"]):
        model.train()
        _slope_mode(True)
        for rows in _batches(rng):
            ids = sorted(set(cplx[rows].tolist()))
            remap = {c: i for i, c in enumerate(ids)}
            # One conformer per complex per epoch, drawn from the epoch RNG.
            # This is data augmentation, not an ensemble: the model sees a
            # different geometry of the same complex each pass, which is what
            # turns 956 structures into ~2,800 and attacks the variance that
            # makes this the noisiest arm in the factorial.  The draw depends
            # only on the complex id and the epoch RNG -- never on the target or
            # on which fold a row belongs to.
            confs = ([int(rng.integers(0, cache.n_conformers(i))) for i in ids]
                     if n_conf > 1 else None)
            b = cache.batch(ids, confs)
            emb = model.encode(b)                      # encode each complex once
            gather = torch.as_tensor([remap[c] for c in cplx[rows]], device=device)
            xpos = np.array([fit_pos[int(r)] for r in rows])
            tab = torch.as_tensor(Xtr[xpos], device=device)
            e = _centre(emb[gather], blk_all[rows])
            if cond_idx and getattr(model, "film", None) is not None:
                e = model.modulate(e, tab[:, cond_idx])
            emb_row = torch.cat([e, tab], -1)
            pred = model.head(emb_row).squeeze(-1)
            st = _slope_terms(emb_row, blk_all[rows])
            if st is not None:
                pred = pred + st
            tgt = torch.as_tensor((y[rows] - ymu) / ysd, device=device)

            if level_w is None:
                # The LEVEL term's robustness, never varied in 462 recorded
                # runs (all Huber, delta 1.0 on the standardised target).
                # Motivated by the one thing that worked in this campaign:
                # swapping CatBoost's RMSE for MAE was worth +0.1066 adjacent
                # AND +0.0115 log D.  The metric is a within-block DIFFERENCE,
                # so a single badly-measured row corrupts every pair it enters;
                # bounding each row's influence is exactly what that buys.
                # Note --pair-loss-kind huber (the PAIR term) did nothing,
                # which localises the leverage to the level fit.
                if level_loss == "mae":
                    loss = nn.functional.l1_loss(pred, tgt)
                elif level_loss == "mse":
                    loss = nn.functional.mse_loss(pred, tgt)
                elif level_loss == "quantile":
                    # Pinball loss.  alpha=0.5 is MAE up to a factor of 2.
                    #
                    # Motivated by a measurement, not by taste: on the tabular
                    # arm Quantile(0.7) scored +0.2384 against MAE/Quantile(0.5)
                    # at +0.2188 and RMSE at +0.1594, and log D is strongly
                    # LEFT-skewed (skew -0.712, mean +0.267 below median +0.352)
                    # because low values sit near detection limits.  An upper
                    # quantile down-weights that untrustworthy tail.
                    #
                    # A constant offset cannot be the mechanism: the metric
                    # scores WITHIN-BLOCK DIFFERENCES, in which any shift
                    # common to a block cancels exactly.
                    e = tgt - pred
                    loss = torch.maximum(level_q * e,
                                         (level_q - 1.0) * e).mean()
                else:
                    loss = nn.functional.huber_loss(pred, tgt,
                                                    delta=level_delta)
            else:
                # Decomposed objective.
                #
                # The scored quantity is a *difference* between two lanthanides
                # inside a block; the block mean is nuisance.  Measured on this
                # dataset, the block mean carries Var 2.41 and the within-block
                # contrast Var 0.25 under the strict key -- so a plain MSE spends
                # **91% of its gradient** on a quantity the metric never looks at,
                # and which CatBoost already predicts better than any net here
                # (overall R2 +0.4987 against the stack's +0.4369).
                #
                # Splitting the loss lets the level and the contrast be weighted
                # independently instead of by whatever ratio their variances
                # happen to have.  ``--pair-loss-weight`` only ever *added* a
                # contrast term on top of the full MSE; it could not take the
                # level term away.
                codes, _ = pd.factorize(blk_all[rows])
                bidx = torch.as_tensor(codes, device=device, dtype=torch.long)
                nB = int(codes.max()) + 1 if len(codes) else 0
                pm, tm = block_means(pred, tgt, bidx, nB)
                # One term per block, not per row: a block with ten measurements
                # is one nuisance parameter, not ten, and row-weighting it would
                # hand the biggest blocks the loss all over again.
                loss = level_w * nn.functional.huber_loss(pm, tm, delta=1.0)

            if aux_head is not None:
                # Masked, not imputed: a complex whose CShM or E_int is missing
                # contributes nothing rather than being taught that it is
                # average.  The head reads the encoder embedding only -- the
                # auxiliary quantity never reaches the log D prediction path,
                # which is the whole point of using it as a target rather than
                # as a feature.
                at = torch.as_tensor(aux_y[rows], device=device)
                ok = torch.isfinite(at).all(dim=-1)
                if bool(ok.any()):
                    ap_ = aux_head(e[ok])
                    loss = loss + aux_w * nn.functional.mse_loss(ap_, at[ok])

            if pair_w > 0:
                # All within-block pairs, weighted towards *adjacent* metals:
                # neighbouring lanthanides are the hardest and the ones the
                # claim is about, but restricting to them alone leaves too few
                # pairs per batch to give a stable gradient.
                cb = blk_all[rows]
                li = lidx_all[rows]
                same = torch.as_tensor(cb[:, None] == cb[None, :], device=device)
                iu = torch.triu(same, diagonal=1)
                pi, pj = torch.nonzero(iu, as_tuple=True)
                if pi.numel() > 0:
                    dl = torch.as_tensor(np.abs(li[:, None] - li[None, :]),
                                         device=device, dtype=torch.float32)[pi, pj]
                    # --- what gets differenced --------------------------------
                    # Published: raw ROW pairs.  The metric never does that.
                    # adjacent_pair_arrays averages y and p within (block, metal)
                    # BEFORE differencing, so a metal measured ten times is one
                    # point, not ten.  Censused on the modelled rows, the cost of
                    # the mismatch is not marginal:
                    #
                    #   dl == 0 (SAME metal, two replicates) 19,482 pairs, and
                    #   because it is weighted 3.0 alongside the adjacent pairs
                    #   and the error is squared, it carries 61.6% of the squared
                    #   mass inside the "adjacent emphasis" term -- the exact
                    #   population evaluation averages away.
                    #   Adjacent row pairs duplicate the metric's 1,349 pairs
                    #   x13.4, and because pair count is quadratic in block size
                    #   while the metric is linear in distinct metals, the ten
                    #   largest blocks take 59.6% of the mass.
                    #
                    # evaluation.py:181-192 already records that enumerating raw
                    # row pairs once produced a figure that INVERTED the
                    # published result.  The evaluator was fixed; this was not.
                    #
                    # --pair-metric-align collapses to (block, metal) cells
                    # first.  Exact in-batch rather than approximate, because
                    # _batches emits WHOLE blocks whenever pair_w > 0: every
                    # replicate of a cell is present, so the in-batch cell mean
                    # IS the metric's cell mean.
                    if pair_align:
                        ccodes, _ = pd.factorize(
                            pd.MultiIndex.from_arrays([cb, li]))
                        nC = int(ccodes.max()) + 1
                        cidx = torch.as_tensor(ccodes, device=device,
                                               dtype=torch.long)
                        # block_means is this reduction already, and is module
                        # level precisely so it can be reused and tested.
                        P, T = block_means(pred, tgt, cidx, nC)
                        first = np.unique(ccodes, return_index=True)[1]
                        cbk, clx = cb[first], li[first]
                        cs = torch.as_tensor(cbk[:, None] == cbk[None, :],
                                             device=device)
                        qi, qj = torch.nonzero(torch.triu(cs, diagonal=1),
                                               as_tuple=True)
                        dql = torch.as_tensor(np.abs(clx[:, None] - clx[None, :]),
                                              device=device,
                                              dtype=torch.float32)[qi, qj]
                    else:
                        P, T, qi, qj, dql = pred, tgt, pi, pj, dl
                    if pair_adj_only:
                        keep = dql <= 1.0
                        qi, qj, dql = qi[keep], qj[keep], dql[keep]
                    if pair_subsample < 1.0 and qi.numel() > 1:
                        # DECISIVE TEST for why --pair-metric-align fails.
                        # Hypothesis: collapsing replicates starves the term
                        # (18,065 row pairs -> ~1,349 cell pairs), the same
                        # data-poverty that killed pair_regressor on 905 pairs.
                        # If starvation is the cause, randomly thinning the
                        # pair set to the SAME count without collapsing must
                        # hurt about as much.  If it does not, the damage is
                        # the collapsing itself and the hypothesis is wrong.
                        k = max(1, int(round(pair_subsample * qi.numel())))
                        sel = torch.as_tensor(
                            rng.choice(qi.numel(), k, replace=False),
                            device=device, dtype=torch.long)
                        qi, qj, dql = qi[sel], qj[sel], dql[sel]
                    if qi.numel() > 0:
                        w = torch.where(dql <= 1.0, pair_adj_w, 1.0)
                        dp = P[qi] - P[qj]
                        dt = T[qi] - T[qj]
                        err = ((dp - dt) ** 2 if pair_kind == "sq"
                               else nn.functional.huber_loss(
                                   dp, dt, delta=pair_delta, reduction="none"))
                        loss = loss + pair_w * (w * err).mean()
                    # T2: the same pairs, but the difference gets its own
                    # parameters instead of being the difference of two scalar
                    # level predictions.  Restricted to ADJACENT metals, which
                    # is exactly the population the metric scores -- the level
                    # surrogate above deliberately includes all within-block
                    # pairs for gradient stability, and this one does not need
                    # to because it is not the only pair signal.
                    if getattr(model, "pair_head", None) is not None:
                        adj = dl <= 1.0
                        if int(adj.sum()) > 0:
                            ai, aj = pi[adj], pj[adj]
                            dhat = model.pair_forward(emb_row, ai, aj)
                            dtrue = tgt[ai] - tgt[aj]
                            loss = loss + ph_w * nn.functional.huber_loss(
                                dhat, dtrue, delta=1.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        if (ep + 1) % cfg["val_every"] == 0:
            vp = _predict(val_idx, Xval)
            if cfg.get("select_on") == "adjacent":
                # Select the checkpoint that best predicts adjacent-pair
                # separation rather than absolute log D.  Computed on the inner
                # validation split only -- those extractants are held out of
                # this fold's training, so no test information is involved.
                m = ev.adjacent_pair_metrics(
                    y[val_idx], vp, blk_all[val_idx], lidx_all[val_idx])
                r2adj = m.get("sel_adj_logSF_r2", float("nan"))
                n_adj = m.get("sel_adj_n_pairs", 0)
                # Fall back to MSE when a fold's validation split happens to
                # contain too few adjacent pairs to score reliably.
                vloss = (-r2adj if np.isfinite(r2adj) and n_adj >= 30
                         else float(np.mean((vp - y[val_idx]) ** 2)))
            else:
                vloss = float(np.mean((vp - y[val_idx]) ** 2))
            if vloss < best_val - 1e-5:
                best_val, patience = vloss, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    if emb_out is not None:
        # Test-row embeddings from the restored best checkpoint -- the same
        # weights that produce the returned predictions, so the embedding and
        # the prediction describe one model, not two.
        model.eval()
        with torch.no_grad():
            for take in _eval_chunks(te_idx):
                rows = te_idx[take]
                ids = sorted(set(cplx[rows].tolist()))
                remap = {c: i for i, c in enumerate(ids)}
                e = _centre(_encode(ids)[torch.as_tensor(
                    [remap[c] for c in cplx[rows]], device=device)],
                    blk_all[rows])
                emb_out[rows] = e.cpu().numpy()
    out = _predict(te_idx, Xte)
    # Any flag the chosen architecture cannot honour must fail here.  DistanceNet
    # does not accept film_dim / angular_readout / attn_pool / node_feat_dim, and
    # train.py builds it with a separate call, so these were accepted, recorded
    # as enabled in results.jsonl, and silently ignored on --arch dist -- the
    # architecture every modern run uses.  No published claim depended on them
    # (checked: zero dist runs set any of them), but a future one would have,
    # and the failure looks exactly like a real null.
    import inspect as _inspect
    _accepts = set(_inspect.signature(type(model).__init__).parameters)
    for _flag, _param in (("film", "film_dim"),
                          ("angular_readout", "angular_readout"),
                          ("attn_pool", "attn_pool"),
                          ("node_angular", "node_feat_dim")):
        if cfg.get(_flag) and _param not in _accepts:
            raise SystemExit(
                f"--{_flag.replace('_', '-')} is not supported by "
                f"{type(model).__name__}; the run would be a silent no-op "
                f"recorded as {_flag}=True.")

    # A requested pair head that the model does not have is a SILENT no-op that
    # still records pair_head=True -- exactly how --arch dist ran four arms to
    # six identical decimal places.  Fail instead.
    if cfg.get("pair_head") and getattr(model, "pair_head", None) is None:
        raise SystemExit("--pair-head was requested but this architecture built "
                         "no pair head; the run would be a silent no-op "
                         "recorded as pair_head=True.")
    if cfg.get("pair_reconcile") and getattr(model, "pair_head", None) is not None:
        out = _reconcile(out, te_idx, Xte)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=("snn", "picnn", "tabular", "dist"),
                    default="snn",
                    help="'tabular' is the no-topology control: identical loop,\n"
                         "loss, folds and seeds with a width-zero embedding.\n"
                         "'dist' is the no-SIMPLEX control: same edges and\n"
                         "readout, continuous-filter messages, no triangles")
    ap.add_argument("--match-rows", choices=("snn", "picnn"), default="snn",
                    help="for --arch tabular, whose row eligibility to reproduce\n"
                         "so the paired bootstrap compares the same rows")
    ap.add_argument("--smoke-epochs", type=int, nargs="+",
                    default=[400, 1500, 4000])
    ap.add_argument("--preset", default="baseline_2d")
    ap.add_argument("--topology-only", action="store_true",
                    help="drop the tabular block entirely (ablation): the\n                          model then sees nothing but the 3D topology")
    ap.add_argument("--tag", default="snn")
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-rows", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--filtration-max", type=float, default=3.5)
    ap.add_argument("--heavy-only", action="store_true", default=True)
    ap.add_argument("--all-atoms", dest="heavy_only", action="store_false")
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select-on", choices=("mse", "adjacent"), default="mse",
                    help="inner-validation criterion for early stopping")
    ap.add_argument("--conformers", type=int, default=1,
                    help="number of geometries per complex to use; >1 samples "
                         "one at random per epoch and mean-pools all of them at "
                         "inference (requires automl/artifacts/vr_conformers)")
    ap.add_argument("--dump-embeddings", action="store_true",
                    help="also save the out-of-fold encoder embeddings, so a "
                         "downstream learner can use the topological "
                         "representation directly")
    ap.add_argument("--block-centre", action="store_true",
                    help="concatenate each embedding with its deviation from "
                         "the composition-block mean, cancelling common-mode "
                         "ligand and conformer noise")
    ap.add_argument("--pair-loss-weight", type=float, default=0.0,
                    help="weight on the within-composition pairwise-difference "
                         "loss; 0 reproduces the plain regression objective")
    # --- CAMPAIGN6: the shape of the contrast term ---------------------------
    # Every default below reproduces the published term byte-for-byte.  The
    # term has never been varied: 391 of 462 recorded runs used weight 2.0, the
    # adjacent emphasis was a literal 3.0 in the source, and no run has ever
    # collapsed replicates before differencing.
    ap.add_argument("--pair-adj-weight", type=float, default=3.0,
                    help="multiplier on |delta lanthanide_index| <= 1 pairs "
                         "inside the contrast term. 3.0 is the value hardcoded "
                         "in every run to date")
    ap.add_argument("--pair-adj-only", action="store_true",
                    help="restrict the contrast term to ADJACENT pairs, the "
                         "population the metric scores. Published behaviour "
                         "keeps all within-block pairs at weight 1 for gradient "
                         "stability; this trades that for exact alignment and "
                         "is much thinner once --pair-metric-align collapses "
                         "replicates")
    ap.add_argument("--pair-loss-kind", choices=("sq", "huber"), default="sq",
                    help="error on the pair difference. 'sq' is published; "
                         "'huber' stops one mis-measured block dominating")
    ap.add_argument("--pair-huber-delta", type=float, default=1.0,
                    help="delta for --pair-loss-kind huber")
    ap.add_argument("--level-quantile", type=float, default=0.5,
                    help="alpha for --level-loss quantile (pinball). 0.5 is "
                         "MAE; the tabular arm peaks ABOVE 0.5 because log D "
                         "has a heavy left tail")
    ap.add_argument("--level-loss", choices=("huber", "mae", "mse", "quantile"),
                    default="huber",
                    help="error on the per-row LEVEL term. 'huber' with "
                         "delta 1.0 is what every run to date used. 'mae' is "
                         "the neural analogue of the CatBoost MAE switch that "
                         "was this campaign's only held-out win")
    ap.add_argument("--level-huber-delta", type=float, default=1.0,
                    help="delta for --level-loss huber; smaller is more "
                         "MAE-like")
    ap.add_argument("--pair-subsample", type=float, default=1.0,
                    help="keep this fraction of within-block pair terms, drawn "
                         "afresh each epoch and WITHOUT collapsing replicates. "
                         "The control for --pair-metric-align: 0.075 matches "
                         "the pair count alignment leaves (1,349 of 18,065), "
                         "so if thinning hurts as much as aligning, the damage "
                         "is data poverty; if it does not, the damage is the "
                         "collapsing itself")
    ap.add_argument("--pair-metric-align", action="store_true",
                    help="average prediction AND target within (block, metal) "
                         "before differencing, exactly as "
                         "evaluation.adjacent_pair_arrays does. Without it, "
                         "61.6%% of the squared mass inside the 3x-weighted "
                         "'adjacent emphasis' term is SAME-METAL replicate "
                         "pairs -- the population evaluation averages away "
                         "before it scores anything")
    ap.add_argument("--level-weight", type=float, default=None,
                    help="turn on the DECOMPOSED objective and set the weight "
                         "on its block-mean (level) term. Unset = the published "
                         "objective: Huber on the raw target, which spends ~91%% "
                         "of its gradient on the block mean the metric never "
                         "looks at. Use with --pair-loss-weight for the contrast "
                         "term, e.g. --level-weight 0.2 --pair-loss-weight 2.0")
    ap.add_argument("--no-triangles", action="store_true",
                    help="drop the 2-simplex level from the SNN, leaving\n                          message passing over the GRAPH of the same complex.\n                          The other half of 'is it simplicial, or just 3D?'")
    ap.add_argument("--node-angular", action="store_true",
                    help="SWEEP2 A2: append a per-node soft histogram of the "
                         "angles its neighbours subtend at it. No angular "
                         "information has ever reached a neural encoder in this "
                         "study (662/662 runs used baseline_2d and scalar node "
                         "inputs); cosines are rotation/translation/reflection "
                         "invariant so this costs no invariance")
    ap.add_argument("--pair-head", action="store_true",
                    help="CAMPAIGN3 T2: a head that predicts the adjacent-pair "
                         "difference DIRECTLY from [h_i, h_j, h_i-h_j], with "
                         "its own parameters, instead of differencing two "
                         "scalar level predictions.")
    ap.add_argument("--pair-head-weight", type=float, default=1.0,
                    help="weight on the T2 pairwise loss")
    ap.add_argument("--geometry", default="full",
                    choices=("full", "shipped", "control", "neutral",
                             "water", "octanol"),
                    help="CAMPAIGN4: which geometry set to train on. 'control' "
                         "is the same complexes re-optimised with no anion; "
                         "'neutral' adds counter-ions to charge-neutralise. "
                         "Each loads its own triangle-edge cache. These are a "
                         "REPLACEMENT set, never mixed within a run.")
    ap.add_argument("--pair-reconcile", action="store_true",
                    help="POST-HOC (campaign 3): at inference, adjust the "
                         "level head's per-(block,metal) means so their "
                         "adjacent differences match the pair head's "
                         "predictions. Without this the pair head's skill "
                         "never reaches the metric, which scores differences "
                         "of LEVEL predictions on a pathway the pair head does "
                         "not touch.")
    ap.add_argument("--film", action="store_true",
                    help="CAMPAIGN3 T3: FiLM the structural embedding on the "
                         "cond__ columns, so 45 diluents and 9 acids reach the "
                         "structure representation instead of only the head.")
    ap.add_argument("--extra-block-mean", action="store_true",
                    help="POST-HOC (not pre-registered): replace every column "
                         "the preset adds beyond baseline_2d by its mean within "
                         "the composition block, removing within-block variation "
                         "while keeping the columns and their between-block "
                         "content. Tests whether A1's collapse is caused by the "
                         "head fitting within-block geometry variation.")
    ap.add_argument("--angular-readout", action="store_true",
                    help="SWEEP2 A3: add a readout block from the donor-M-donor "
                         "angle distribution -- the coordination polyhedron "
                         "itself, which CShM and bite angles only summarise")
    ap.add_argument("--attn-pool", action="store_true",
                    help="SWEEP2 C2: attention pooling with the metal embedding "
                         "as query. No attention of any kind exists in this repo")
    ap.add_argument("--radial-bins", type=int, default=None,
                    help="SWEEP2 C1: radial shell histogram resolution "
                         "(hardcoded 32 in every run to date)")
    ap.add_argument("--radial-max", type=float, default=None,
                    help="SWEEP2 C1: radial shell histogram range in Angstrom "
                         "(hardcoded 8.0 in every run to date)")
    ap.add_argument("--aux-target", default=None,
                    choices=("cshm", "eint", "qtransfer"),
                    help="SWEEP2 axis B: train a SECOND head on a physical "
                         "quantity alongside log D. Never attempted in this "
                         "study. Reuses the energy campaign that failed as "
                         "INPUTS -- as inputs they substituted for the exact "
                         "ionic radius and destroyed selectivity; as targets "
                         "they can only shape the representation")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="weight on the auxiliary loss")
    ap.add_argument("--radius-slope", choices=("off", "linear", "quad", "basis"),
                    default="off",
                    help="fit pred = f(u) + sum_k g_k(u)*phi_k(metal) with u "
                         "made metal-free, so the predicted within-block "
                         "adjacent difference is exactly sum_k g_k(u)*d phi_k. "
                         "'linear' K=1 (radius), 'quad' K=2 (+radius^2, the "
                         "cavity-mismatch term), 'basis' K=4 (+lanthanide "
                         "index and its square, which give the basis a "
                         "coordinate not collinear with radius). Unrelated to "
                         "the g13__slope FEATURE block in dataset.py")
    ap.add_argument("--radius-slope-u", choices=("block", "row"),
                    default="block",
                    help="how u is made metal-free. 'block' also replaces the "
                         "structural half by its composition-block mean, which "
                         "is what makes the difference identity hold; 'row' "
                         "zeroes only the four metal COLUMNS and is NOT exact "
                         "under --arch snn/dist, where the embedding is a "
                         "different complex per metal")
    ap.add_argument("--edge-asset", default=None,
                    help="load the neighbour graph from "
                         "automl/artifacts/vr_cutoff/<NAME> instead of the "
                         "shipped 4.0 A Vietoris-Rips asset. Carries NO "
                         "triangles, so it requires --arch dist or "
                         "--no-triangles. NOTE --filtration-max still "
                         "thresholds edges at load, so leaving it at 3.5 "
                         "discards everything the rebuild added")
    ap.add_argument("--rbf-bins", type=int, default=None,
                    help="--arch dist: resolution of the Gaussian radial basis "
                         "on the EDGE distance (hardcoded 32 in every run)")
    ap.add_argument("--rbf-max", type=float, default=None,
                    help="--arch dist: range of that basis, in Angstrom. Unset "
                         "= --filtration-max. This widens the BASIS only; the "
                         "receptive field is set by --filtration-max")
    ap.add_argument("--block-key", default=None,
                    choices=("composition_key", "strict_composition_key"),
                    help="which blocking the contrast loss, the block-centred "
                         "embedding and adjacent-pair checkpoint selection use. "
                         "Unset = composition_key, the published choice")
    ap.add_argument("--restrict-groups", default=None,
                    help="file of extractant names, one per line; the run sees "
                         "only those. Used to keep a hyperparameter sweep off "
                         "the confirmation half of the frozen split.")
    ap.add_argument("--smoke", action="store_true",
                    help="overfit a tiny subset; sanity check that it can learn")
    ap.add_argument("--deterministic", action="store_true",
                    help="fix the reduction order so a re-run reproduces "
                         "bit-for-bit; see the note below on why this matters")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    if args.deterministic:
        # Why this exists.
        #
        # PI_SWEEP_PRECISION.md measured an 8-seed ensemble moving by 0.0092
        # between two runs of the *identical* configuration, and showed that
        # more seeds does not fix it (8 seeds bought a factor of 1.76 where
        # independence would give 2.83) because part of the noise is shared
        # across every seed within a process.  That floor is larger than most of
        # the differences this study argues about: it is why re-running one cell
        # of 25 changed Stage A's winner, and why a sweep could not select.
        #
        # Three separate sources, all switched off here:
        #   * scatter atomics -- the dominant one for the SNN, which has no
        #     convolutions at all;  snn.set_deterministic swaps the sum
        #     reductions for sort-based ones.
        #   * cuDNN autotuning -- picks a different algorithm per process, which
        #     is the shared-across-seeds component for the persistence-image CNN.
        #   * cuBLAS workspace reuse -- needs an environment variable, and it is
        #     only read when the cuBLAS handle is created, so setting it after
        #     CUDA is up is too late.  The SLURM scripts export it; setting it
        #     here as well is a belt-and-braces fallback for interactive runs.
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from automl.topo import snn as _snn
        _snn.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: index_reduce_ (segment max) has no deterministic CUDA
        # kernel, but a maximum is order-independent anyway, so raising on it
        # would block a run that is in fact reproducible.  See snn.scatter_max.
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[topo] deterministic mode: sorted scatter, cudnn.deterministic, "
              "benchmark off", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[topo] device={device} torch={torch.__version__}", flush=True)

    if args.arch == "tabular" and args.topology_only:
        # Width-zero embedding AND width-zero design matrix: a model with no
        # inputs at all, which would train to the target mean and be recorded
        # as though it were a control.
        raise SystemExit("--arch tabular --topology-only leaves the model no "
                         "inputs; use --arch snn/picnn --topology-only for the "
                         "topology-only ablation.")
    if args.radius_slope != "off":
        if args.topology_only:
            raise SystemExit("--radius-slope needs Ionic Radius_metal from the "
                             "tabular block; --topology-only removes it.")
        if args.pair_reconcile:
            # Both set the adjacent increments the metric reads, and
            # _reconcile overwrites them wholesale -- so the arm would measure
            # neither head.
            raise SystemExit("--radius-slope and --pair-reconcile both set the "
                             "adjacent increments; --pair-reconcile would "
                             "discard the interaction head's.")
        if args.pair_head:
            print("[topo] WARNING: --radius-slope and --pair-head both regress "
                  "the adjacent difference, and pair_forward reads emb_row, "
                  "which does not carry the interaction term. Crossing them "
                  "double-counts.", flush=True)
    if args.edge_asset:
        if args.arch != "dist" and not args.no_triangles:
            raise SystemExit(
                "--edge-asset carries no 2-simplices; an --arch snn run "
                "without --no-triangles would be a no-triangles model recorded "
                "as the full one. Use --arch dist or --no-triangles.")
        if args.geometry != "full":
            raise SystemExit("--edge-asset and --geometry are both REPLACEMENT "
                             "assets; combining them is undefined")
        if args.conformers > 1:
            raise SystemExit("--edge-asset has no conformer axis")
        if args.filtration_max < 4.0:
            print(f"[topo] WARNING: --edge-asset with --filtration-max "
                  f"{args.filtration_max} thresholds the rebuilt edges back "
                  f"down; you almost certainly want --filtration-max >= the "
                  f"build cutoff", flush=True)
    df, X, cols = build_row_table(args.preset, args.arch, args.match_rows,
                                  geometry=args.geometry,
                                  edge_asset_name=args.edge_asset)
    if args.extra_block_mean:
        # POST-HOC diagnostic for sweep2 A1, not part of the pre-registration.
        #
        # Replace every column the preset adds beyond baseline_2d by its mean
        # within the composition block.  The columns, their count and their
        # between-block content are unchanged; only the within-block variation
        # is removed.  So this isolates the one thing the adjacent-pair metric
        # can be damaged by, holding the size of the feature addition fixed.
        #
        # Not a leak: composition_key = extractant || conditions and the CV
        # groups by extractant, so no block spans a fold boundary (checked:
        # 0 of 552 blocks span more than one extractant_group).  A block mean
        # taken over the whole table is therefore identical to one taken inside
        # the fold, and no label is involved either way.
        from automl.dataset import BLOCK_PRESETS
        base = set(build_row_table("baseline_2d", args.arch, args.match_rows)[2])
        tgt = [i for i, c in enumerate(cols) if c not in base]
        if not tgt:
            raise SystemExit("--extra-block-mean needs a preset that adds "
                             "columns beyond baseline_2d")
        blk = df["composition_key"].to_numpy()
        order = np.argsort(blk, kind="stable")
        starts = np.concatenate(([0], np.flatnonzero(blk[order][1:] != blk[order][:-1]) + 1))
        for seg in np.split(order, starts[1:]):
            sub = X[np.ix_(seg, tgt)]
            with np.errstate(invalid="ignore"):
                m = np.nanmean(sub, axis=0)
            X[np.ix_(seg, tgt)] = np.where(np.isfinite(m), m, np.nan)
        print(f"[topo] --extra-block-mean: {len(tgt)} added columns replaced by "
              f"their per-block means over {len(starts)} blocks", flush=True)
    if args.restrict_groups:
        # Confine the run to a named set of extractants.  This is what keeps a
        # hyperparameter sweep off the confirmation half: selection is made on
        # the tune extractants only, so the confirm extractants never influence
        # which configuration is chosen and the confirmatory interval needs no
        # penalty for the number of configurations tried.
        #
        # Applied here, before folds are built, so the held-out extractants are
        # absent from training, from the inner early-stopping split and from the
        # out-of-fold table alike -- not merely filtered out at scoring time.
        want = {ln.strip() for ln in
                Path(args.restrict_groups).read_text().splitlines() if ln.strip()}
        have = set(df[GROUP_COL].astype(str))
        unknown = want - have
        if unknown:
            raise SystemExit(f"--restrict-groups names {len(unknown)} extractants "
                             f"not in the data, e.g. {sorted(unknown)[:3]}")
        keep = df[GROUP_COL].astype(str).isin(want).to_numpy()
        if not keep.any():
            raise SystemExit("--restrict-groups selects no rows")
        print(f"[topo] restrict-groups: {keep.sum()}/{len(df)} rows, "
              f"{len(want)} of {len(have)} extractants "
              f"({Path(args.restrict_groups).name})", flush=True)
        df = df[keep].reset_index(drop=True)
        X = X[keep]
    if args.topology_only:
        # A zero-width design matrix, not a zeroed one: this sets tabular_dim=0
        # so the head has no tabular weights at all.  Passing zeros instead
        # would leave those weights present and trainable on a constant, which
        # is a different (and weaker) ablation than the one being claimed.
        X = X[:, :0]
        cols = []
    print(f"[topo] rows={len(df)} extractants={df[GROUP_COL].nunique()} "
          f"distinct complexes={df['_cplx'].nunique()} tabular_dim={X.shape[1]}",
          flush=True)

    if args.arch == "tabular":
        # No asset is loaded, but the harness still batches by distinct complex
        # and gathers per row, so the control's sampling matches the arm it
        # controls for exactly.
        cache = NullCache(device)
        n_assets = int(df["_cplx"].nunique())
    elif args.arch == "picnn":
        P = PersistenceImages()
        cache = ImageCache(P, device)
        n_assets = len(P)
        # 1 for the shipped images (H0+H1 summed); 2 when a swept configuration
        # renders the homology dimensions as separate channels.
        pi_channels = P.n_channels
        print(f"[topo] persistence images: {P.images.shape} from {PI_PATH.name}",
              flush=True)
    else:
        # ConformerComplexes is a superset wrapper: conformer 0 is the shipped
        # geometry and index_of/__len__ match SimplicialComplexes exactly, so
        # the row set is identical whichever is loaded.
        if args.edge_asset:
            S = edge_asset(args.edge_asset)
            mp = EDGE_ROOT / args.edge_asset / "meta.json"
            info = json.loads(mp.read_text()) if mp.exists() else {}
            print(f"[topo] edge asset '{args.edge_asset}': {len(S)} complexes, "
                  f"{info.get('n_edges', '?')} edges, mean degree "
                  f"{info.get('mean_degree', '?')}", flush=True)
        elif args.geometry != "full":
            S = geometry_asset(args.geometry)
            print(f"[topo] geometry set '{args.geometry}': {len(S)} complexes",
                  flush=True)
        else:
            S = (ConformerComplexes(verbose=False) if args.conformers > 1
                 else SimplicialComplexes(verbose=False))
        # Independently, so each axis-A cell changes exactly one thing.
        cache = ComplexCache(S, args.filtration_max, args.heavy_only, device,
                             node_angular=bool(args.node_angular),
                             metal_angular=bool(args.angular_readout))
        n_assets = len(S)
        if args.conformers > 1:
            tot = sum(S.n_conformers(k) for k in range(len(S)))
            print(f"[topo] conformers: {tot} structures over {len(S)} complexes "
                  f"({tot/max(len(S),1):.2f} per complex)", flush=True)
    cfg = {k: getattr(args, k) for k in
           ("dim", "layers", "dropout", "head_hidden", "lr", "weight_decay",
            "epochs", "batch_rows", "eval_batch", "val_every", "patience")}
    cfg["arch"] = args.arch
    cfg["pair_loss_weight"] = args.pair_loss_weight
    cfg["pair_adj_weight"] = args.pair_adj_weight
    cfg["pair_adj_only"] = args.pair_adj_only
    cfg["pair_loss_kind"] = args.pair_loss_kind
    cfg["pair_huber_delta"] = args.pair_huber_delta
    cfg["pair_metric_align"] = args.pair_metric_align
    cfg["pair_subsample"] = args.pair_subsample
    cfg["level_loss"] = args.level_loss
    cfg["level_huber_delta"] = args.level_huber_delta
    cfg["level_quantile"] = args.level_quantile
    cfg["select_on"] = args.select_on
    cfg["level_weight"] = args.level_weight
    cfg["block_key"] = args.block_key
    cfg["no_triangles"] = args.no_triangles
    cfg["extra_block_mean"] = args.extra_block_mean
    cfg["_cols"] = list(cols)          # for FiLM's cond__ lookup; not a knob
    cfg["pair_head"] = args.pair_head
    cfg["pair_head_weight"] = args.pair_head_weight
    cfg["film"] = args.film
    cfg["pair_reconcile"] = args.pair_reconcile
    cfg["geometry"] = args.geometry
    cfg["node_angular"] = args.node_angular
    cfg["angular_readout"] = args.angular_readout
    cfg["attn_pool"] = args.attn_pool
    cfg["radius_slope"] = args.radius_slope
    cfg["radius_slope_u"] = args.radius_slope_u
    cfg["edge_asset"] = args.edge_asset
    cfg["rbf_bins"] = args.rbf_bins
    cfg["rbf_max"] = args.rbf_max
    cfg["radial_bins"] = args.radial_bins
    cfg["radial_max"] = args.radial_max
    cfg["aux_target"] = args.aux_target
    cfg["aux_weight"] = args.aux_weight if args.aux_target else None
    cfg["n_conformers"] = args.conformers
    cfg["block_centre"] = args.block_centre
    if args.arch == "picnn":
        cfg["in_channels"] = pi_channels
        cfg["pi_images"] = str(PI_PATH)

    if args.smoke:
        # Capacity check, not a generalisation check: regularisation OFF.
        # Leaving dropout and weight decay on here measures the wrong thing --
        # a model that *can* fit will still look like it cannot.
        # One row per complex.  df.head(60) spans only 21 distinct geometries,
        # and rows sharing a geometry differ only in conditions -- so a
        # topology-only encoder is capped at R2 = 0.884 there no matter how
        # good it is.  Testing capacity against an unreachable ceiling measures
        # the sampling, not the architecture.
        sub = df.drop_duplicates("_cplx").head(60).index.to_numpy()
        base = {**cfg, "patience": 10_000, "val_every": 10_000,
                "dropout": 0.0, "weight_decay": 0.0, "lr": 3e-3,
                "batch_rows": 30, "no_inner_val": True}
        y_sub = df[TARGET].to_numpy()[sub]
        Xz = np.zeros_like(X)
        # A curve, not a single number: if R2 is still climbing with steps the
        # limit is optimisation (fixable with schedule/epochs); if it plateaus
        # the limit is capacity (fixable only with a bigger encoder).  One
        # number cannot tell those apart, and they need opposite fixes.
        best = 0.0
        for ep in args.smoke_epochs:
            c = {**base, "epochs": ep}
            r2 = ev._r2(y_sub, run_fold(df, X, cache, sub, sub, cfg=c,
                                        device=device, seed=args.seed))
            r2t = ev._r2(y_sub, run_fold(df, Xz, cache, sub, sub, cfg=c,
                                         device=device, seed=args.seed))
            best = max(best, r2)
            # For the tabular control the second run zeroes the *only* inputs
            # the model has, so it is a different and equally useful check: it
            # must collapse to ~0.  A tabular arm that still fits with its
            # features blanked is reading something it should not.
            other = ("blanked-features" if args.arch == "tabular"
                     else "topology-only")
            print(f"[smoke] epochs={ep:5d}  hybrid R2={r2:.4f}   "
                  f"{other} R2={r2t:.4f}", flush=True)
        print(f"[smoke] best hybrid R2 = {best:.4f} (regularisation off; "
              f"must exceed 0.95 to show the architecture can fit at all)")
        return 0 if best > 0.95 else 1

    pretrained = None
    if args.pretrain_epochs and args.arch == "snn":
        # The pretrained encoder is computed once and shared across seeds.
        #
        # It is self-supervised -- masked charges and edge radii, no log D -- so
        # it is the *same* optimisation problem for every seed, and recomputing
        # it per run cost an hour each (~16 h across 32 seeds at 2 concurrent
        # nodes) to arrive at approximately the same weights.
        #
        # This does not weaken the ensembling lever.  S0 has no pretraining at
        # all and still shows a +0.060 ensemble gain with per-seed SD 0.047, so
        # all of its seed diversity already comes from the supervised phase --
        # fold splits, batch order, dropout, head init -- none of which a shared
        # encoder touches.
        #
        # The cache key carries every parameter that changes what the weights
        # mean.  Reusing a cache built at a different width, depth or conformer
        # count would silently warm-start from the wrong encoder.
        key = (f"d{args.dim}_l{args.layers}_e{args.pretrain_epochs}"
               f"_c{args.conformers}_s{PRETRAIN_SEED}")
        cache_path = PRETRAIN_DIR / f"encoder_{key}.pt"
        model = SimplicialNet(dim=args.dim, layers=args.layers,
                              dropout=args.dropout, tabular_dim=0).to(device)
        if cache_path.exists():
            model.load_state_dict(torch.load(cache_path, map_location=device))
            print(f"[topo] loaded pretrained encoder from {cache_path.name}",
                  flush=True)
        else:
            n_struct = (sum(cache.n_conformers(k) for k in range(n_assets))
                        if args.conformers > 1 else n_assets)
            print(f"[topo] pretraining encoder ({count_parameters(model)} "
                  f"params) over {n_struct} structures "
                  f"({n_assets} complexes)", flush=True)
            pretrain(model, cache, list(range(n_assets)),
                     epochs=args.pretrain_epochs, batch_size=16, lr=1e-3,
                     device=device, seed=PRETRAIN_SEED)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.replace(cache_path)      # atomic: concurrent array tasks race
            print(f"[topo] cached pretrained encoder -> {cache_path.name}",
                  flush=True)
        pretrained = {k: v.detach().clone() for k, v in model.state_dict().items()}

    groups = df[GROUP_COL].to_numpy()
    y = df[TARGET].to_numpy(dtype=float)
    oof_sum = np.zeros(len(df)); oof_cnt = np.zeros(len(df))
    emb_sum = emb_cnt = None       # allocated lazily: width is known per model
    t0 = time.time()
    for rep in range(args.repeats):
        for fi, (tr, te) in enumerate(
                ev.grouped_folds(groups, n_splits=args.folds, seed=args.seed + rep)):
            ts = time.time()
            emb_buf = None
            if args.dump_embeddings:
                width = (2 if args.block_centre else 1) * 9 * args.dim
                if emb_sum is None:
                    emb_sum = np.zeros((len(df), width), dtype=np.float32)
                    emb_cnt = np.zeros(len(df), dtype=np.float32)
                emb_buf = np.zeros((len(df), width), dtype=np.float32)
            pred = run_fold(df, X, cache, tr, te, cfg=cfg, device=device,
                            seed=args.seed + rep * 100 + fi,
                            pretrained_state=pretrained, emb_out=emb_buf)
            oof_sum[te] += pred; oof_cnt[te] += 1
            if emb_buf is not None:
                emb_sum[te] += emb_buf[te]; emb_cnt[te] += 1
            print(f"  rep{rep} fold{fi}: n_te={len(te)} "
                  f"R2={ev._r2(y[te], pred):+.4f} [{time.time()-ts:.0f}s]",
                  flush=True)
    oof = oof_sum / np.maximum(oof_cnt, 1)

    metrics = ev.full_metrics(y, oof, df)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # The filtration/heavy suffix describes a Vietoris-Rips construction that a
    # tabular run never performed; carrying it would label the control as though
    # it had used one.
    stem = (f"{args.tag}_{args.arch}_{args.preset}"
            + ("" if args.arch == "tabular"
               else f"_f{args.filtration_max}_h{int(args.heavy_only)}")
            # the ablation must not overwrite the full model's OOF file
            + ("_notri" if args.no_triangles else "")
            + ("_ph" if args.pair_head else "")
            + ("_film" if args.film else "")
            + ("_rec" if args.pair_reconcile else "")
            + ("" if args.geometry == "full" else f"_{args.geometry}")
            + ("_xbm" if args.extra_block_mean else "")
            + ("_nang" if args.node_angular else "")
            + ("_arod" if args.angular_readout else "")
            + ("_attn" if args.attn_pool else "")
            + (f"_aux{args.aux_target}" if args.aux_target else "")
            + (f"_rb{args.radial_bins}" if args.radial_bins else "")
            + (f"_rm{args.radial_max}" if args.radial_max else "")
            # CAMPAIGN6.  Empty for every published configuration, so no
            # existing stem moves and no campaign run can overwrite one.
            + ("_pal" if args.pair_metric_align else "")
            + ("" if args.pair_subsample == 1.0
               else f"_ps{args.pair_subsample}")
            + ("" if args.level_loss == "huber" else f"_ll{args.level_loss}")
            + (f"_lq{args.level_quantile}"
               if args.level_loss == "quantile" else "")
            + ("" if args.level_huber_delta == 1.0
               else f"_ld{args.level_huber_delta}")
            + ("" if args.pair_adj_weight == 3.0
               else f"_paw{args.pair_adj_weight}")
            + ("_pao" if args.pair_adj_only else "")
            + ("" if args.pair_loss_kind == "sq"
               else f"_pk{args.pair_loss_kind}")
            + ("" if args.radius_slope == "off" else f"_rs{args.radius_slope}")
            + ("_rsurow" if args.radius_slope != "off"
               and args.radius_slope_u == "row" else "")
            + (f"_ea{args.edge_asset}" if args.edge_asset else "")
            + (f"_fb{args.rbf_bins}" if args.rbf_bins else "")
            + (f"_fm{args.rbf_max}" if args.rbf_max else ""))
    pd.DataFrame({
        "safe_exp_id": df["safe_exp_id"].to_numpy(), "y": y, "oof": oof,
        "extractant_group": groups,
        "composition_key": df["composition_key"].to_numpy(),
        "metal": df["metal"].to_numpy(),
        "lanthanide_index": df["lanthanide_index"].to_numpy(),
    }).to_parquet(out_dir / f"oof_{stem}.parquet", index=False)
    # ``cfg`` is recorded alongside ``vars(args)`` because it carries what the
    # command line does not: the resolved persistence-image asset and its
    # channel count, both set by the PI_IMAGES_PATH environment override.  The
    # sweep analysis identifies runs by recorded configuration rather than by
    # tag, so this is what makes a run self-describing.
    rec = {"tag": args.tag, "preset": args.preset, "config": vars(args),
           "resolved": {k: v for k, v in cfg.items()
                        if k in ("arch", "in_channels", "pi_images",
                                 "pair_loss_weight", "pair_adj_weight",
                                 "pair_adj_only", "pair_loss_kind",
                                 "pair_huber_delta", "pair_metric_align",
                                 "pair_subsample",
                                 "level_loss", "level_huber_delta",
                                 "level_quantile",
                                 "select_on",
                                 "level_weight", "block_key",
                                 "no_triangles", "pair_head",
                                 "pair_head_weight", "film",
                                 "pair_reconcile", "geometry",
                                 "extra_block_mean",
                                 "node_angular",
                                 "angular_readout", "attn_pool",
                                 "radial_bins", "radial_max",
                                 "rbf_bins", "rbf_max", "edge_asset",
                                 "radius_slope", "radius_slope_u",
                                 "aux_target", "aux_weight")},
           "n_rows": int(len(df)), "n_complexes": int(df["_cplx"].nunique()),
           "metrics": {k: (None if not np.isfinite(v) else float(v))
                       for k, v in metrics.items()},
           "seconds": time.time() - t0}
    if emb_sum is not None:
        # Averaged over repeats, exactly like the predictions, so the embedding
        # matrix and the OOF vector describe the same ensemble.
        emb = emb_sum / np.maximum(emb_cnt, 1)[:, None]
        np.savez_compressed(out_dir / f"emb_{stem}.npz",
                            embeddings=emb.astype(np.float32),
                            safe_exp_id=df["safe_exp_id"].to_numpy().astype("U64"))
        print(f"[topo] wrote embeddings {emb.shape} -> emb_{stem}.npz", flush=True)
    (out_dir / f"run_{stem}.json").write_text(json.dumps(rec, indent=2))
    with open(out_dir / "results.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    print("\n" + ev.format_metrics(metrics))
    print(f"  adjacent-pair logSF R2 = {metrics.get('sel_adj_logSF_r2', float('nan')):.4f} "
          f"(n={metrics.get('sel_adj_n_pairs', 0):.0f}), "
          f"sign acc = {metrics.get('sel_adj_sign_accuracy', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
