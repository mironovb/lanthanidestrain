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
from automl.topo.pi_cnn import PersistenceImages, PersistenceCNN
from automl.topo.tabular_net import TabularNet, NullCache

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "automl/artifacts/topo_runs"
PRETRAIN_DIR = REPO / "automl/artifacts/pretrained"
# Fixed, and deliberately independent of the run seed: the pretrained encoder is
# shared across seeds, so tying it to one of them would be misleading about
# which run produced it.
PRETRAIN_SEED = 42


# ---------------------------------------------------------------------------
def build_row_table(preset: str = "baseline_2d", arch: str = "snn",
                    match_rows: str = "snn"
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
    key_arch = match_rows if arch == "tabular" else arch
    asset = (SimplicialComplexes(verbose=False) if key_arch == "snn"
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


def _standardise(train_X, *others):
    if train_X.shape[1] == 0:                      # topology-only ablation
        return [train_X] + [o for o in others]
    med = np.nanmedian(train_X, axis=0)
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

    def batch(self, ids: list[int]):
        return self.P.batch(ids, self.device)


class ComplexCache:
    """Pre-loads and caches collated complexes on the target device."""

    def __init__(self, S: SimplicialComplexes, filtration_max, heavy_only,
                 device):
        self.S, self.device = S, device
        self.filtration_max, self.heavy_only = filtration_max, heavy_only
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
                                          heavy_only=self.heavy_only)
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
def run_fold(df, X, cache, tr_idx, te_idx, *, cfg, device, seed,
             pretrained_state=None, target_col: str | None = None) -> np.ndarray:
    """Train on tr_idx, predict te_idx.  Returns predictions for te_idx."""
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
    y = df[target_col or TARGET].to_numpy(dtype=np.float32)
    ymu, ysd = float(y[fit_idx].mean()), float(y[fit_idx].std() or 1.0)

    if cfg.get("arch", "snn") == "tabular":
        model = TabularNet(dim=cfg["dim"], dropout=cfg["dropout"],
                           tabular_dim=X.shape[1],
                           head_hidden=cfg["head_hidden"]).to(device)
    elif cfg.get("arch", "snn") == "picnn":
        model = PersistenceCNN(dim=cfg["dim"], dropout=cfg["dropout"],
                               tabular_dim=X.shape[1],
                               head_hidden=cfg["head_hidden"]).to(device)
    else:
        model = SimplicialNet(dim=cfg["dim"], layers=cfg["layers"],
                              dropout=cfg["dropout"], tabular_dim=X.shape[1],
                              head_hidden=cfg["head_hidden"],
                              head_embed_mult=2 if cfg.get("block_centre") else 1
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
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
    pair_w = float(cfg.get("pair_loss_weight", 0.0))
    fit_blocks: list[np.ndarray] = []
    if pair_w > 0:
        pos = {int(r): i for i, r in enumerate(fit_idx)}
        by_block: dict[Any, list[int]] = {}
        for r in fit_idx:
            by_block.setdefault(comp_all[r], []).append(int(r))
        fit_blocks = [np.array(v) for v in by_block.values() if len(v) >= 2]

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
        if not block_centre:
            step = max(cfg["eval_batch"], 1)
            return [np.arange(s, min(s + step, len(idx)))
                    for s in range(0, len(idx), step)]
        budget = max(cfg["eval_batch"], 1)
        chunks, buf = [], []
        for _key, positions in pd.Series(comp_all[idx]).groupby(
                comp_all[idx], sort=False).groups.items():
            pos = np.asarray(positions, dtype=int)
            if buf and len(buf) + len(pos) > budget:
                chunks.append(np.concatenate(buf)); buf = []
            buf.append(pos)
        if buf:
            chunks.append(np.concatenate(buf))
        return chunks

    def _predict(idx, Xs):
        model.eval()
        outs = np.empty(len(idx), dtype=np.float64)
        with torch.no_grad():
            for take in _eval_chunks(idx):
                rows = idx[take]
                ids = sorted(set(cplx[rows].tolist()))
                remap = {c: i for i, c in enumerate(ids)}
                emb = _encode(ids)
                gather = torch.as_tensor([remap[c] for c in cplx[rows]],
                                         device=device)
                e = _centre(emb[gather], comp_all[rows])
                tab = torch.as_tensor(Xs[take], device=device)
                outs[take] = (model.head(torch.cat([e, tab], -1))
                              .squeeze(-1).cpu().numpy())
        return outs * ysd + ymu

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

    for ep in range(cfg["epochs"]):
        model.train()
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
            e = _centre(emb[gather], comp_all[rows])
            pred = model.head(torch.cat([e, tab], -1)).squeeze(-1)
            tgt = torch.as_tensor((y[rows] - ymu) / ysd, device=device)
            loss = nn.functional.huber_loss(pred, tgt, delta=1.0)

            if pair_w > 0:
                # All within-block pairs, weighted towards *adjacent* metals:
                # neighbouring lanthanides are the hardest and the ones the
                # claim is about, but restricting to them alone leaves too few
                # pairs per batch to give a stable gradient.
                cb = comp_all[rows]
                li = lidx_all[rows]
                same = torch.as_tensor(cb[:, None] == cb[None, :], device=device)
                iu = torch.triu(same, diagonal=1)
                pi, pj = torch.nonzero(iu, as_tuple=True)
                if pi.numel() > 0:
                    dl = torch.as_tensor(np.abs(li[:, None] - li[None, :]),
                                         device=device, dtype=torch.float32)[pi, pj]
                    w = torch.where(dl <= 1.0, 3.0, 1.0)      # emphasise neighbours
                    dp = pred[pi] - pred[pj]
                    dt = tgt[pi] - tgt[pj]
                    loss = loss + pair_w * (w * (dp - dt) ** 2).mean()
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
                    y[val_idx], vp, comp_all[val_idx], lidx_all[val_idx])
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
    return _predict(te_idx, Xte)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=("snn", "picnn", "tabular"), default="snn",
                    help="'tabular' is the no-topology control: identical loop,\n"
                         "loss, folds and seeds with a width-zero embedding")
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
    ap.add_argument("--block-centre", action="store_true",
                    help="concatenate each embedding with its deviation from "
                         "the composition-block mean, cancelling common-mode "
                         "ligand and conformer noise")
    ap.add_argument("--pair-loss-weight", type=float, default=0.0,
                    help="weight on the within-composition pairwise-difference "
                         "loss; 0 reproduces the plain regression objective")
    ap.add_argument("--smoke", action="store_true",
                    help="overfit a tiny subset; sanity check that it can learn")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[topo] device={device} torch={torch.__version__}", flush=True)

    if args.arch == "tabular" and args.topology_only:
        # Width-zero embedding AND width-zero design matrix: a model with no
        # inputs at all, which would train to the target mean and be recorded
        # as though it were a control.
        raise SystemExit("--arch tabular --topology-only leaves the model no "
                         "inputs; use --arch snn/picnn --topology-only for the "
                         "topology-only ablation.")
    df, X, cols = build_row_table(args.preset, args.arch, args.match_rows)
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
    else:
        # ConformerComplexes is a superset wrapper: conformer 0 is the shipped
        # geometry and index_of/__len__ match SimplicialComplexes exactly, so
        # the row set is identical whichever is loaded.
        S = (ConformerComplexes(verbose=False) if args.conformers > 1
             else SimplicialComplexes(verbose=False))
        cache = ComplexCache(S, args.filtration_max, args.heavy_only, device)
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
    cfg["select_on"] = args.select_on
    cfg["n_conformers"] = args.conformers
    cfg["block_centre"] = args.block_centre

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
    t0 = time.time()
    for rep in range(args.repeats):
        for fi, (tr, te) in enumerate(
                ev.grouped_folds(groups, n_splits=args.folds, seed=args.seed + rep)):
            ts = time.time()
            pred = run_fold(df, X, cache, tr, te, cfg=cfg, device=device,
                            seed=args.seed + rep * 100 + fi,
                            pretrained_state=pretrained)
            oof_sum[te] += pred; oof_cnt[te] += 1
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
               else f"_f{args.filtration_max}_h{int(args.heavy_only)}"))
    pd.DataFrame({
        "safe_exp_id": df["safe_exp_id"].to_numpy(), "y": y, "oof": oof,
        "extractant_group": groups,
        "composition_key": df["composition_key"].to_numpy(),
        "metal": df["metal"].to_numpy(),
        "lanthanide_index": df["lanthanide_index"].to_numpy(),
    }).to_parquet(out_dir / f"oof_{stem}.parquet", index=False)
    rec = {"tag": args.tag, "preset": args.preset, "config": vars(args),
           "n_rows": int(len(df)), "n_complexes": int(df["_cplx"].nunique()),
           "metrics": {k: (None if not np.isfinite(v) else float(v))
                       for k, v in metrics.items()},
           "seconds": time.time() - t0}
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
