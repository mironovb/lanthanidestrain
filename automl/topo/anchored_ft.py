#!/usr/bin/env python3
"""A tabular transformer (FT-Transformer) in the anchored level/shape harness.

The question: does a transformer over the 746 tabular columns beat gradient
boosting as the level model, the shape model, or both?  Evaluation is the
one every other number in this project uses: leave-extractants-out, 5 folds
x 3 repeats, out-of-fold, scored on adjacent-pair log SF R^2.

Model (Gorishniy et al. 2021, adapted for the sparse fingerprint block):
  * each of the 81 numeric columns (RDKit, metal, conditions, plan) becomes
    its own token through a per-feature linear embedding;
  * the 665 ECFP bits are folded into 8 tokens by a linear layer per group of
    bits -- one token per bit would make 746-wide attention and buys nothing
    for binary indicators;
  * a CLS token, ``layers`` pre-norm transformer blocks, a head on CLS.
Losses mirror the CatBoost champion: pinball loss at q = 0.6 for the level
model and for the shape model (the anch_q60_q60 cell), so a difference in
score is a difference in learner, not in objective.

Cells:
  ft_flat   level model only (the transformer analogue of flat CatBoost)
  ft_anch   level + shape both transformers (analogue of anch_q60_q60)
The per-row level and shape predictions are stored separately so hybrid
systems (CatBoost anchor + transformer shape, and vice versa) can be
assembled without retraining.

Writes automl/artifacts/anchored_ft/oof_<cell>_s<seed>.parquet
(safe_exp_id, y, oof, level, shape) and automl/reports/anchored_ft.csv.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.anchored_ft --cells ft_anch \
            --seeds 42 51 67 83 [--population ok_only|has3d]
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
from automl.matrix_cache import load_cache

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/anchored_ft"
OUT_CSV = REPO / "automl/reports/anchored_ft.csv"


# --------------------------------------------------------------------------
class FTTransformer(nn.Module):
    def __init__(self, n_numeric: int, n_bits: int, dim: int = 64,
                 layers: int = 3, heads: int = 8, dropout: float = 0.1,
                 bit_tokens: int = 8):
        super().__init__()
        self.n_numeric, self.n_bits = n_numeric, n_bits
        self.num_w = nn.Parameter(torch.randn(n_numeric, dim) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_numeric, dim))
        self.bit_tokens = bit_tokens
        # bits are split into `bit_tokens` contiguous groups; each group is
        # linearly embedded into one token
        self.group = int(np.ceil(n_bits / bit_tokens)) if n_bits else 0
        self.bit_emb = nn.ModuleList([
            nn.Linear(min(self.group, n_bits - g * self.group), dim)
            for g in range(bit_tokens)]) if n_bits else nn.ModuleList()
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=2 * dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.ReLU(),
                                  nn.Linear(dim, 1))

    def forward(self, x_num: torch.Tensor, x_bits: torch.Tensor):
        toks = [self.cls.expand(x_num.shape[0], -1, -1),
                x_num.unsqueeze(-1) * self.num_w + self.num_b]
        for g, lin in enumerate(self.bit_emb):
            toks.append(lin(x_bits[:, g * self.group:(g + 1) * self.group])
                        .unsqueeze(1))
        h = self.blocks(torch.cat(toks, dim=1))
        return self.head(h[:, 0]).squeeze(-1)


def pinball(pred, y, q):
    e = y - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


# --------------------------------------------------------------------------
def load_table(population: str):
    df, blocks, _ = load_cache()
    if population == "ok_only":
        df = df[df["geometry_ok"].astype(bool) & df["has_3d"]]
    elif population == "collab":
        cflag = pd.read_parquet(
            REPO / "collaborator_update/dataset.parquet",
            columns=["safe_exp_id", "geometry_ok"]
        ).set_index("safe_exp_id")["geometry_ok"]
        df = df[df["safe_exp_id"].map(cflag).fillna(False).astype(bool)]
    else:
        df = df[df["has_3d"].astype(bool)]
    df = df.reset_index(drop=True)
    num_cols, bit_cols = [], []
    for b in ("rdkit", "metal", "cond", "plan"):
        num_cols.extend(c for c in blocks.mapping[b] if c in df.columns)
    bit_cols = [c for c in blocks.mapping["ecfp"] if c in df.columns]
    num_cols = list(dict.fromkeys(num_cols))
    return df, df[num_cols].to_numpy(np.float32), \
        df[bit_cols].to_numpy(np.float32)


def fit_predict(Xn, Xb, y, tr, te, q, seed, device, epochs=150, dim=64,
                layers=3, heads=8, dropout=0.1, lr=1e-3, wd=1e-4, bs=128,
                patience=8, val_frac=0.15, groups=None):
    """Train on tr with an inner extractant-grouped validation split for
    early stopping; return predictions on te (in original units)."""
    rng = np.random.default_rng(seed)
    g_tr = groups[tr]
    uniq = np.unique(g_tr)
    vg = set(rng.choice(uniq, size=max(1, int(val_frac * len(uniq))),
                        replace=False).tolist())
    is_val = np.array([g in vg for g in g_tr])
    fit, val = tr[~is_val], tr[is_val]

    med = np.nanmedian(Xn[fit], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def prep_n(a):
        a = np.where(np.isfinite(a), a, med)
        return a
    Xn_fit = prep_n(Xn[fit]); mu = Xn_fit.mean(0); sd = Xn_fit.std(0) + 1e-6
    def std_n(a): return ((prep_n(a) - mu) / sd).astype(np.float32)
    ymu, ysd = float(y[fit].mean()), float(y[fit].std() or 1.0)

    torch.manual_seed(seed)
    model = FTTransformer(Xn.shape[1], Xb.shape[1], dim, layers, heads,
                          dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    T = lambda a: torch.as_tensor(a, device=device)
    Xn_f, Xb_f, y_f = T(std_n(Xn[fit])), T(Xb[fit]), T(((y[fit] - ymu) / ysd)
                                                      .astype(np.float32))
    Xn_v, Xb_v, y_v = T(std_n(Xn[val])), T(Xb[val]), T(((y[val] - ymu) / ysd)
                                                      .astype(np.float32))
    best, best_state, bad = np.inf, None, 0
    n = len(fit)
    for ep in range(epochs):
        model.train()
        perm = torch.as_tensor(rng.permutation(n), device=device)
        for s0 in range(0, n, bs):
            idx = perm[s0:s0 + bs]
            opt.zero_grad()
            loss = pinball(model(Xn_f[idx], Xb_f[idx]), y_f[idx], q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 2 == 1:
            model.eval()
            with torch.no_grad():
                vl = float(pinball(model(Xn_v, Xb_v), y_v, q))
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
    with torch.no_grad():
        out = []
        Xn_t, Xb_t = T(std_n(Xn[te])), T(Xb[te])
        for s0 in range(0, len(te), 512):
            out.append(model(Xn_t[s0:s0 + 512], Xb_t[s0:s0 + 512]).cpu().numpy())
    return np.concatenate(out) * ysd + ymu


def run_cell(name, df, Xn, Xb, seed, device, shape: bool, q=0.6,
             folds=5, repeats=3, **hp):
    y = df["log_D"].to_numpy(np.float32)
    g = df["extractant_group"].to_numpy()
    comp = df["composition_key"].to_numpy()
    level = np.zeros(len(y)); shp = np.zeros(len(y)); cnt = np.zeros(len(y))
    t0 = time.time()
    for rep in range(repeats):
        for k, (tr, te) in enumerate(ev.grouped_folds(g, folds, seed=seed + rep)):
            lv = fit_predict(Xn, Xb, y, tr, te, q, seed + rep, device,
                             groups=g, **hp)
            level[te] += lv
            if shape:
                key_tr = pd.Series(g[tr])
                resid = y[tr] - key_tr.map(
                    pd.Series(y[tr]).groupby(key_tr).mean()).to_numpy()
                sv = fit_predict(Xn, Xb, resid.astype(np.float32), tr, te, q,
                                 seed + rep + 1000, device, groups=g, **hp)
                shp[te] += sv
            cnt[te] += 1
            print(f"    [{name} s{seed}] rep {rep} fold {k} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    level /= np.maximum(cnt, 1); shp /= np.maximum(cnt, 1)
    if shape:
        # anchored combination at shape_weight 1: anchor from the level model,
        # shape centred within the block
        bp, sp = pd.Series(level), pd.Series(shp)
        key = pd.Series(g)
        anchor = bp.groupby(key).transform("mean")
        shape_c = sp - sp.groupby(key).transform("mean")
        oof = (anchor + shape_c).to_numpy()
    else:
        oof = level
    dy, dp = ev.adjacent_pair_arrays(y, oof, comp,
                                     df["lanthanide_index"].to_numpy())
    res = {"cell": f"{name}_s{seed}", "adj_r2": ev._r2(dy, dp),
           "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
           "adj_disp": float(np.std(dp) / np.std(dy)),
           "logD_r2": ev._r2(y, oof), "n_pairs": int(len(dy))}
    ART.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y, "oof": oof,
                  "level": level, "shape": shp}).to_parquet(
        ART / f"oof_{name}_s{seed}.parquet", index=False)
    return res


CELLS = {"ft_flat": dict(shape=False), "ft_anch": dict(shape=True)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=["ft_anch"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--population", default="ok_only",
                    choices=("ok_only", "has3d", "collab"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, Xn, Xb = load_table(args.population)
    print(f"{len(df)} rows · {df['extractant_group'].nunique()} extractants · "
          f"{Xn.shape[1]} numeric + {Xb.shape[1]} bits · {device}", flush=True)
    pop = "" if args.population == "ok_only" else f"_{args.population}"
    rows = []
    for cell in args.cells:
        oofs = []
        for sd in args.seeds:
            res = run_cell(cell + pop, df, Xn, Xb, sd, device,
                           repeats=args.repeats, epochs=args.epochs,
                           dim=args.dim, layers=args.layers, **CELLS[cell])
            rows.append(res)
            print(f"  {res['cell']:24s} adj_R2={res['adj_r2']:+.4f} "
                  f"P2={res['adj_pearson2']:+.4f} disp={res['adj_disp']:.3f} "
                  f"logD_R2={res['logD_r2']:+.4f}", flush=True)
            oofs.append(pd.read_parquet(ART / f"oof_{cell + pop}_s{sd}.parquet"))
        if len(oofs) > 1:
            ens = oofs[0][["safe_exp_id", "y"]].copy()
            for c in ("oof", "level", "shape"):
                ens[c] = np.mean([o[c].to_numpy() for o in oofs], axis=0)
            ens.to_parquet(ART / f"oof_{cell + pop}_ens{len(oofs)}.parquet",
                           index=False)
            dy, dp = ev.adjacent_pair_arrays(
                ens["y"].to_numpy(), ens["oof"].to_numpy(),
                df["composition_key"].to_numpy(),
                df["lanthanide_index"].to_numpy())
            r = {"cell": f"{cell + pop}_ens{len(oofs)}",
                 "adj_r2": ev._r2(dy, dp),
                 "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
                 "adj_disp": float(np.std(dp) / np.std(dy)),
                 "logD_r2": ev._r2(ens["y"].to_numpy(), ens["oof"].to_numpy()),
                 "n_pairs": int(len(dy))}
            rows.append(r)
            print(f"  {r['cell']:24s} adj_R2={r['adj_r2']:+.4f} "
                  f"P2={r['adj_pearson2']:+.4f}", flush=True)
    out = pd.DataFrame(rows)
    if OUT_CSV.exists():
        out = pd.concat([pd.read_csv(OUT_CSV), out], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
