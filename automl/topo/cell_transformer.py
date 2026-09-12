#!/usr/bin/env python3
"""A set transformer over the (block, metal) CELLS of a composition block.

The adjacent-pair metric averages replicate rows of one metal inside a
composition block into a single cell and then differences neighbouring cells
(``evaluation.adjacent_pair_arrays``).  ``block_transformer`` tokenised rows,
so a block measured under many conditions became a long, padded sequence in
which attention was spread over condition replicates rather than metals.
This version tokenises the cells the metric actually scores: one token per
(block, metal), carrying the metal descriptors, the mean conditions over the
cell's rows and the row count, plus a block-level ligand context added to
every token.  The output is centred within the block by construction and
trained with an L1 loss against the block-centred cell-mean log D.

Only blocks with at least two metals carry a training signal (205 of 552 in
the legacy population); single-metal blocks are neither trained on nor
scored.  Everything else follows the anchored protocol: leave-extractants-out
5 folds x 3 repeats, inner extractant-grouped early stopping, deterministic
seeds, per-row out-of-fold shape written to
automl/artifacts/cell_tf/oof_celltf<pop>_s<seed>.parquet (safe_exp_id, y,
shape) and a summary row in automl/reports/cell_transformer.csv.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.cell_transformer --seeds 42 51
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
from automl.topo.anchored_champion import load_cache
from automl.topo.anchored_ft import load_table

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/cell_tf"
OUT_CSV = REPO / "automl/reports/cell_transformer.csv"


class CellTransformer(nn.Module):
    def __init__(self, n_tok: int, n_ctx: int, dim: int = 64, layers: int = 2,
                 heads: int = 4, dropout: float = 0.2, n_metal: int = 16):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(n_tok, dim), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(dim, dim))
        self.ctx = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_ctx, dim),
                                 nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(dim, dim))
        self.metal = nn.Embedding(n_metal, dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, num_layers=layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, X, C, midx, mask):
        """X (B, T, n_tok) cell features, C (B, n_ctx) block context, midx
        (B, T) long, mask (B, T) True where a cell exists.  Returns
        block-centred predictions (B, T)."""
        h = self.tok(X) + self.metal(midx) + self.ctx(C).unsqueeze(1)
        h = self.blocks(h, src_key_padding_mask=~mask)
        p = self.out(h).squeeze(-1).masked_fill(~mask, 0.0)
        n = mask.sum(1, keepdim=True).clamp(min=1)
        return (p - p.sum(1, keepdim=True) / n) * mask


def build_cells(df: pd.DataFrame, Xn: np.ndarray, Xb: np.ndarray,
                num_cols: list[str], blocks_map):
    """Aggregate rows into (block, metal) cells.

    Returns cell frame (block, metal, n, first row), token features (cells x
    n_tok), block context (blocks x n_ctx), and per-row cell / block ids.
    """
    ctx_cols = [c for c in blocks_map["rdkit"] if c in num_cols]
    tok_cols = [c for c in num_cols if c not in ctx_cols]     # metal, cond, plan
    ci = [num_cols.index(c) for c in ctx_cols]
    ti = [num_cols.index(c) for c in tok_cols]
    key = pd.Series(list(zip(df["composition_key"], df["lanthanide_index"])))
    cell_id = key.astype("category").cat.codes.to_numpy()
    n_cells = int(cell_id.max()) + 1
    Xt = np.full((n_cells, len(ti) + 1), np.nan, np.float32)
    meta = {}
    for c in range(n_cells):
        rows = np.flatnonzero(cell_id == c)
        with np.errstate(all="ignore"):
            Xt[c, :-1] = np.nanmean(Xn[rows][:, ti], axis=0)
        Xt[c, -1] = np.log1p(len(rows))
        meta[c] = (df["composition_key"].iat[rows[0]],
                   int(df["lanthanide_index"].iat[rows[0]]), len(rows), rows[0])
    cells = pd.DataFrame.from_dict(meta, orient="index",
                                   columns=["block", "metal", "n", "row0"])
    block_id = cells["block"].astype("category").cat.codes.to_numpy()
    n_blocks = int(block_id.max()) + 1
    Xc = np.zeros((n_blocks, len(ci) + Xb.shape[1]), np.float32)
    for b in range(n_blocks):
        rows = np.flatnonzero(np.isin(cell_id, np.flatnonzero(block_id == b)))
        with np.errstate(all="ignore"):
            Xc[b, :len(ci)] = np.nanmean(Xn[rows][:, ci], axis=0)
        Xc[b, len(ci):] = Xb[rows].mean(0)
    return cells, block_id, Xt, Xc, cell_id


def batch_tensors(bl, Xt, Xc, yc, metal, device):
    T = max(len(cs) for _, cs in bl); B = len(bl)
    X = np.zeros((B, T, Xt.shape[1]), np.float32); y = np.zeros((B, T), np.float32)
    m = np.zeros((B, T), np.int64); k = np.zeros((B, T), bool)
    C = np.zeros((B, Xc.shape[1]), np.float32)
    for i, (b, cs) in enumerate(bl):
        X[i, :len(cs)] = Xt[cs]; y[i, :len(cs)] = yc[cs]
        m[i, :len(cs)] = metal[cs]; k[i, :len(cs)] = True; C[i] = Xc[b]
    t = lambda a: torch.as_tensor(a, device=device)
    return t(X), t(C), t(y), t(m), t(k)


def fit_predict(Xt, Xc, ycell, block_id, metal, cell_group, tr_cells, te_cells,
                seed, device, epochs=300, dim=64, layers=2, heads=4,
                dropout=0.2, lr=1e-3, wd=1e-3, blocks_per_batch=32,
                patience=30, val_frac=0.15):
    rng = np.random.default_rng(seed)
    uniq = np.unique(cell_group[tr_cells])
    vg = set(rng.choice(uniq, size=max(1, int(val_frac * len(uniq))),
                        replace=False).tolist())
    is_val = np.array([g in vg for g in cell_group[tr_cells]])
    fit, val = tr_cells[~is_val], tr_cells[is_val]

    def groups(cells):
        d = {}
        for c in cells:
            d.setdefault(int(block_id[c]), []).append(int(c))
        return [(b, np.array(v)) for b, v in d.items() if len(v) >= 2]
    fit_bl, val_bl, te_bl = groups(fit), groups(val), groups(te_cells)

    # standardise on the fitted cells / blocks; impute by their medians
    def prep(A, idx):
        med = np.nanmedian(A[idx], axis=0); med = np.where(np.isfinite(med), med, 0)
        Ai = np.where(np.isfinite(A), A, med)
        mu, sd = Ai[idx].mean(0), Ai[idx].std(0) + 1e-6
        return ((Ai - mu) / sd).astype(np.float32)
    Xts = prep(Xt, fit)
    Xcs = prep(Xc, np.unique(block_id[fit]))
    # block-centred cell target
    yc = np.zeros_like(ycell, dtype=np.float32)
    for b in np.unique(block_id):
        cs = np.flatnonzero(block_id == b); yc[cs] = ycell[cs] - ycell[cs].mean()
    fit_cells_multi = np.concatenate([cs for _, cs in fit_bl]) if fit_bl else fit
    ysd = float(yc[fit_cells_multi].std() or 1.0); yc = yc / ysd

    torch.manual_seed(seed)
    model = CellTransformer(Xt.shape[1], Xc.shape[1], dim, layers, heads,
                            dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    V = batch_tensors(val_bl, Xts, Xcs, yc, metal, device) if val_bl else None

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(fit_bl))
        for s0 in range(0, len(order), blocks_per_batch):
            bl = [fit_bl[i] for i in order[s0:s0 + blocks_per_batch]]
            X, C, y, m, k = batch_tensors(bl, Xts, Xcs, yc, metal, device)
            opt.zero_grad()
            p = model(X, C, m, k)
            loss = (torch.abs(p - y) * k).sum() / k.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if V is not None:
            model.eval()
            with torch.no_grad():
                X, C, y, m, k = V
                vl = float((torch.abs(model(X, C, m, k) - y) * k).sum() / k.sum())
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {a: v.detach().clone()
                              for a, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    out = np.zeros(len(ycell), np.float32)
    with torch.no_grad():
        for s0 in range(0, len(te_bl), 128):
            bl = te_bl[s0:s0 + 128]
            X, C, _, m, k = batch_tensors(bl, Xts, Xcs, yc, metal, device)
            p = model(X, C, m, k).cpu().numpy()
            for i, (_, cs) in enumerate(bl):
                out[cs] = p[i, :len(cs)]
    return out[te_cells] * ysd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--population", default="ok_only",
                    choices=("ok_only", "has3d", "collab"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="celltf")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, Xn, Xb = load_table(args.population)
    _, blocks_map, _ = load_cache()
    num_cols = []
    for b in ("rdkit", "metal", "cond", "plan"):
        num_cols.extend(c for c in blocks_map.mapping[b] if c in df.columns)
    num_cols = list(dict.fromkeys(num_cols))
    y = df["log_D"].to_numpy(np.float32)
    comp = df["composition_key"].to_numpy()
    g = df["extractant_group"].to_numpy()
    midx = df["lanthanide_index"].to_numpy().astype(int)
    cells, block_id, Xt, Xc, cell_id = build_cells(df, Xn, Xb, num_cols,
                                                   blocks_map.mapping)
    ycell = np.array([y[cell_id == c].mean() for c in range(len(cells))],
                     np.float32)
    metal = cells["metal"].to_numpy()
    cell_group = np.array([g[r] for r in cells["row0"]])
    multi = pd.Series(block_id).map(pd.Series(block_id).value_counts()) >= 2
    print(f"{len(df)} rows -> {len(cells)} cells in {block_id.max()+1} blocks "
          f"({int(multi.sum())} cells in multi-metal blocks) · tokens "
          f"{Xt.shape[1]} · context {Xc.shape[1]} · {device}", flush=True)
    pop = "" if args.population == "ok_only" else f"_{args.population}"
    hp = dict(epochs=args.epochs, dim=args.dim, layers=args.layers,
              dropout=args.dropout, lr=args.lr)

    rows, oofs = [], []
    for sd in args.seeds:
        shp = np.zeros(len(cells)); cnt = np.zeros(len(cells)); t0 = time.time()
        for rep in range(args.repeats):
            for k, (tr, te) in enumerate(ev.grouped_folds(g, 5, seed=sd + rep)):
                in_te = np.zeros(len(y), bool); in_te[te] = True
                te_cells = np.flatnonzero(pd.Series(in_te).groupby(cell_id).all())
                tr_cells = np.flatnonzero(~pd.Series(in_te).groupby(cell_id).any())
                shp[te_cells] += fit_predict(Xt, Xc, ycell, block_id, metal,
                                             cell_group, tr_cells, te_cells,
                                             sd + rep, device, **hp)
                cnt[te_cells] += 1
            print(f"    [s{sd}] rep {rep} done ({time.time()-t0:.0f}s)",
                  flush=True)
        shp /= np.maximum(cnt, 1)
        shp_rows = shp[cell_id]
        dy, dp = ev.adjacent_pair_arrays(y, shp_rows, comp, midx)
        res = {"cell": f"{args.tag}{pop}_s{sd}", "adj_r2": ev._r2(dy, dp),
               "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
               "adj_disp": float(np.std(dp) / np.std(dy)), "n_pairs": int(len(dy)),
               **{k: v for k, v in hp.items()}}
        rows.append(res)
        print(f"  {res['cell']:20s} adj_R2={res['adj_r2']:+.4f} "
              f"P2={res['adj_pearson2']:+.4f} disp={res['adj_disp']:.3f}",
              flush=True)
        ART.mkdir(parents=True, exist_ok=True)
        o = pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y,
                          "shape": shp_rows})
        o.to_parquet(ART / f"oof_{args.tag}{pop}_s{sd}.parquet", index=False)
        oofs.append(o)
    if len(oofs) > 1:
        ens = oofs[0][["safe_exp_id", "y"]].copy()
        ens["shape"] = np.mean([o["shape"].to_numpy() for o in oofs], axis=0)
        ens.to_parquet(ART / f"oof_{args.tag}{pop}_ens{len(oofs)}.parquet",
                       index=False)
        dy, dp = ev.adjacent_pair_arrays(y, ens["shape"].to_numpy(), comp, midx)
        r = {"cell": f"{args.tag}{pop}_ens{len(oofs)}", "adj_r2": ev._r2(dy, dp),
             "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
             "adj_disp": float(np.std(dp) / np.std(dy)), "n_pairs": int(len(dy)),
             **hp}
        rows.append(r)
        print(f"  {r['cell']:20s} adj_R2={r['adj_r2']:+.4f} "
              f"P2={r['adj_pearson2']:+.4f}", flush=True)
    out = pd.DataFrame(rows)
    if OUT_CSV.exists():
        out = pd.concat([pd.read_csv(OUT_CSV), out], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
