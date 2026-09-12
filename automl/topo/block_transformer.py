#!/usr/bin/env python3
"""A transformer over the lanthanide series inside a block, as the shape model.

The scored quantity is a within-block difference between metals, and the
shape model of the anchored system predicts each row on its own.  This
module tries the architecture whose inductive bias matches the target: the
rows of one block (one extractant, one condition set, up to 14 metals) are
the tokens of a set transformer; every metal attends to every other metal of
its block; the output is centred within the block by construction and
trained against the block-centred log D with an L1 loss.

It replaces only the shape term.  The anchor is taken from the CatBoost
level model already on disk (block mean of the anch_q60_q60 8-seed
predictions), and the score is the adjacent-pair R^2 -- which depends on the
shape term alone, since the anchor cancels in every pair.

Protocol: leave-extractants-out 5 folds x 3 repeats (blocks never straddle
folds because a block belongs to one extractant), inner extractant-grouped
early stopping, deterministic seeds.  Writes
automl/artifacts/block_tf/oof_blocktf_s<seed>.parquet (safe_exp_id, y, shape)
and automl/reports/block_transformer.csv.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.block_transformer --seeds 42 51
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import automl.evaluation as ev
from automl.topo.anchored_ft import load_table

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/block_tf"
OUT_CSV = REPO / "automl/reports/block_transformer.csv"
ANCHOR = REPO / "automl/artifacts/anchored_champ/oof_anch_q60_q60_ens8.parquet"


class BlockTransformer(nn.Module):
    def __init__(self, n_in: int, dim: int = 96, layers: int = 2,
                 heads: int = 4, dropout: float = 0.15, n_metal: int = 16):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, dim), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(dim, dim))
        self.metal = nn.Embedding(n_metal, dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, X, midx, mask):
        """X (B, T, n_in), midx (B, T) long, mask (B, T) True where a row
        exists.  Returns block-centred predictions (B, T)."""
        h = self.inp(X) + self.metal(midx)
        h = self.blocks(h, src_key_padding_mask=~mask)
        p = self.out(h).squeeze(-1)
        p = p.masked_fill(~mask, 0.0)
        n = mask.sum(1, keepdim=True).clamp(min=1)
        return (p - (p.sum(1, keepdim=True) / n)) * mask


def make_blocks(idx: np.ndarray, comp: np.ndarray):
    """Group row indices by composition block."""
    d = {}
    for i in idx:
        d.setdefault(comp[i], []).append(int(i))
    return [np.array(v) for v in d.values()]


def batch_tensors(blocks, X, y_c, midx, device):
    T = max(len(b) for b in blocks)
    B = len(blocks)
    Xb = np.zeros((B, T, X.shape[1]), np.float32)
    yb = np.zeros((B, T), np.float32)
    mb = np.zeros((B, T), np.int64)
    mask = np.zeros((B, T), bool)
    for i, b in enumerate(blocks):
        Xb[i, :len(b)] = X[b]; yb[i, :len(b)] = y_c[b]
        mb[i, :len(b)] = midx[b]; mask[i, :len(b)] = True
    t = lambda a: torch.as_tensor(a, device=device)
    return t(Xb), t(yb), t(mb), t(mask)


def fit_predict(X, y, comp, midx, groups, tr, te, seed, device, epochs=200,
                dim=96, layers=2, heads=4, dropout=0.15, lr=5e-4, wd=1e-4,
                blocks_per_batch=32, patience=10, val_frac=0.15):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups[tr])
    vg = set(rng.choice(uniq, size=max(1, int(val_frac * len(uniq))),
                        replace=False).tolist())
    is_val = np.array([g in vg for g in groups[tr]])
    fit, val = tr[~is_val], tr[is_val]

    med = np.nanmedian(X[fit], axis=0); med = np.where(np.isfinite(med), med, 0)
    Xi = np.where(np.isfinite(X), X, med)
    mu, sd = Xi[fit].mean(0), Xi[fit].std(0) + 1e-6
    Xs = ((Xi - mu) / sd).astype(np.float32)
    # block-centred target, on the training rows' own blocks
    yc = np.zeros_like(y, dtype=np.float32)
    for b in make_blocks(np.arange(len(y)), comp):
        yc[b] = y[b] - y[b].mean()
    ysd = float(yc[fit].std() or 1.0)
    yc = yc / ysd

    fit_blocks = [b for b in make_blocks(fit, comp) if len(b) >= 2]
    val_blocks = [b for b in make_blocks(val, comp) if len(b) >= 2]
    torch.manual_seed(seed)
    model = BlockTransformer(X.shape[1], dim, layers, heads, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xv, yv, mv, kv = batch_tensors(val_blocks, Xs, yc, midx, device) \
        if val_blocks else (None,) * 4

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(fit_blocks))
        for s0 in range(0, len(order), blocks_per_batch):
            bl = [fit_blocks[i] for i in order[s0:s0 + blocks_per_batch]]
            Xb, yb, mb, mk = batch_tensors(bl, Xs, yc, midx, device)
            opt.zero_grad()
            p = model(Xb, mb, mk)
            loss = (torch.abs(p - yb) * mk).sum() / mk.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if Xv is not None and ep % 2 == 1:
            model.eval()
            with torch.no_grad():
                pv = model(Xv, mv, kv)
                vl = float((torch.abs(pv - yv) * kv).sum() / kv.sum())
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    out = np.zeros(len(y), np.float32)
    te_blocks = make_blocks(te, comp)
    with torch.no_grad():
        for s0 in range(0, len(te_blocks), 64):
            bl = te_blocks[s0:s0 + 64]
            Xb, _, mb, mk = batch_tensors(bl, Xs, yc, midx, device)
            p = model(Xb, mb, mk).cpu().numpy()
            for i, b in enumerate(bl):
                out[b] = p[i, :len(b)]
    return out[te] * ysd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--population", default="ok_only",
                    choices=("ok_only", "has3d", "collab"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, Xn, Xb = load_table(args.population)
    X = np.hstack([Xn, Xb])
    y = df["log_D"].to_numpy(np.float32)
    comp = df["composition_key"].to_numpy()
    g = df["extractant_group"].to_numpy()
    midx = df["lanthanide_index"].to_numpy().astype(int)
    print(f"{len(df)} rows · {len(set(comp))} blocks · {X.shape[1]} features "
          f"· {device}", flush=True)
    pop = "" if args.population == "ok_only" else f"_{args.population}"

    rows, oofs = [], []
    for sd in args.seeds:
        shp = np.zeros(len(y)); cnt = np.zeros(len(y)); t0 = time.time()
        for rep in range(args.repeats):
            for k, (tr, te) in enumerate(ev.grouped_folds(g, 5, seed=sd + rep)):
                shp[te] += fit_predict(X, y, comp, midx, g, tr, te,
                                       sd + rep, device, epochs=args.epochs)
                cnt[te] += 1
                print(f"    [s{sd}] rep {rep} fold {k} ({time.time()-t0:.0f}s)",
                      flush=True)
        shp /= np.maximum(cnt, 1)
        dy, dp = ev.adjacent_pair_arrays(y, shp, comp, midx)
        res = {"cell": f"blocktf{pop}_s{sd}", "adj_r2": ev._r2(dy, dp),
               "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
               "adj_disp": float(np.std(dp) / np.std(dy)), "n_pairs": int(len(dy))}
        rows.append(res)
        print(f"  {res['cell']:20s} adj_R2={res['adj_r2']:+.4f} "
              f"P2={res['adj_pearson2']:+.4f} disp={res['adj_disp']:.3f}",
              flush=True)
        ART.mkdir(parents=True, exist_ok=True)
        o = pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y, "shape": shp})
        o.to_parquet(ART / f"oof_blocktf{pop}_s{sd}.parquet", index=False)
        oofs.append(o)
    if len(oofs) > 1:
        ens = oofs[0][["safe_exp_id", "y"]].copy()
        ens["shape"] = np.mean([o["shape"].to_numpy() for o in oofs], axis=0)
        ens.to_parquet(ART / f"oof_blocktf{pop}_ens{len(oofs)}.parquet",
                       index=False)
        dy, dp = ev.adjacent_pair_arrays(y, ens["shape"].to_numpy(), comp, midx)
        r = {"cell": f"blocktf{pop}_ens{len(oofs)}", "adj_r2": ev._r2(dy, dp),
             "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
             "adj_disp": float(np.std(dp) / np.std(dy)), "n_pairs": int(len(dy))}
        rows.append(r)
        print(f"  {r['cell']:20s} adj_R2={r['adj_r2']:+.4f}", flush=True)
    out = pd.DataFrame(rows)
    if OUT_CSV.exists():
        out = pd.concat([pd.read_csv(OUT_CSV), out], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
