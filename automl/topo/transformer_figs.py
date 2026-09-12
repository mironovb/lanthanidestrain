#!/usr/bin/env python3
"""Figures for the transformer experiments (legacy 905 pairs).

(a) per-seed standalone adjacent-pair R2 of every 3D / tabular shape source,
    with its seed ensemble, against the distance-encoder reference;
(b) the blend score as a function of the source weight in the anchored
    system, pair-weighted R2 (what the metric scores) and the
    equal-extractant MSE (what the nested weight minimises).

Reads automl/reports/transformer_eval.json and the per-seed parquets;
writes automl/reports/figures/transformers_legacy.png.  All verdict strings
are computed from the numbers they describe.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from automl.topo.transformer_eval import ART, OUT, SOURCES, per_seed_scores

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "automl/reports/figures/transformers_legacy.png"

LABEL = {"dist": "distance\nencoder",
         "dist8": "distance enc.\n8 matched seeds",
         "attn": "attention\nencoder\ndense", "attn_sp": "attention\nencoder\n<= 4 A",
         "ft": "FT-Transf.\nlevel/shape", "blocktf": "block transf.\nrows",
         "celltf": "block transf.\ncells"}
COLOR = {"dist": "#4a4a4a", "dist8": "#8c8c8c", "attn": "#1f77b4",
         "attn_sp": "#6baed6", "ft": "#d62728", "blocktf": "#ff7f0e",
         "celltf": "#e6a23c"}


def dist_per_seed() -> list[float]:
    rows = [json.loads(l) for l in open(ART / "topo_c15/results.jsonl")]
    return [r["metrics"]["sel_adj_logSF_r2"] for r in rows
            if str(r["tag"]).startswith("c15_plw4")]


def main() -> int:
    res = json.loads(OUT.read_text())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.98, top=0.86, bottom=0.2)

    # (a) per-seed strip + ensemble
    order = ["dist", "attn", "attn_sp", "ft", "blocktf", "celltf"]
    seeds = {"dist": dist_per_seed()}
    for k in order[1:]:
        seeds[k] = per_seed_scores(*SOURCES[k])
    ens = {k: res["standalone"][k]["r2"] for k in order}
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        v = np.asarray(seeds[k])
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=22, alpha=0.55,
                   color=COLOR[k], edgecolor="none")
        ax.plot([i - 0.3, i + 0.3], [ens[k], ens[k]], color=COLOR[k], lw=2.6)
        ax.text(i, ens[k] + 0.012, f"ens {ens[k]:+.3f}\n(n={len(v)})",
                ha="center", va="bottom", fontsize=8.5, color=COLOR[k])
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[k] for k in order], fontsize=8.5)
    ax.set_ylabel("adjacent-pair log SF R$^2$, 905 pairs (standalone)")
    d_mean, d_sd = np.mean(seeds["dist"]), np.std(seeds["dist"], ddof=1)
    a_mean, a_sd = np.mean(seeds["attn"]), np.std(seeds["attn"], ddof=1)
    ax.set_title(f"(a) per seed: attention {a_mean:+.3f} ± {a_sd:.3f} vs "
                 f"distance {d_mean:+.3f} ± {d_sd:.3f}", fontsize=10.5)
    ax.set_ylim(min(-0.42, min(min(v) for v in seeds.values()) - 0.03), 0.34)

    # (b) weight curves
    wc = res["w_curve"]; w = wc["w"]
    for k in ["dist", "dist8", "attn", "attn_sp", "ft", "blocktf", "celltf"]:
        if k not in wc["sources"]:
            continue
        r = wc["sources"][k]["r2"]
        bx.plot(w, r, marker="o", ms=4, lw=1.8, color=COLOR[k],
                label=LABEL[k].replace("\n", " "))
    tab = res["tabular_only"]["r2"]
    bx.axhline(tab, color="#999", lw=0.9, ls="--")
    bx.text(1.0, tab + 0.002, f"tabular only {tab:+.4f}", ha="right",
            va="bottom", fontsize=8.5, color="#666")
    bx.axvline(0.35, color="#bbb", lw=0.8, ls=":")
    bx.set_xlabel("weight w of the source in  anchor + (1-w)·tabular shape + w·source shape\n"
                  "(dotted line: w = 0.35, the fixed weight of the current best system)")
    bx.set_ylabel("adjacent-pair log SF R$^2$, 905 pairs")
    best = {k: (w[int(np.argmax(v["r2"]))], max(v["r2"]))
            for k, v in wc["sources"].items()}
    bx.set_title(f"(b) blend: attention peaks {best['attn'][1]:+.4f} at w {best['attn'][0]:.1f}, "
                 f"distance {best['dist'][1]:+.4f} at w {best['dist'][0]:.1f}", fontsize=10.5)
    bx.legend(fontsize=8, loc="lower left", frameon=False)
    bx.set_xlim(-0.02, 1.02)

    n_hurt = sum(1 for k in ("ft", "blocktf", "celltf")
                 if k in wc["sources"] and all(
                     r <= tab + 1e-9 for r in wc["sources"][k]["r2"]))
    fig.suptitle("Transformer sources in the anchored level/shape system, legacy population "
                 f"(leave-extractants-out, out-of-fold); {n_hurt} of 3 tabular transformers "
                 "lower the score at every weight", fontsize=11)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=170)
    print(f"wrote {FIG}")
    confirm_figure()
    return 0


CONFIRM = REPO / "automl/reports/transformer_confirm.json"
FIG2 = REPO / "automl/reports/figures/transformers_confirm.png"


def confirm_figure() -> None:
    """The one pre-registered look: attention substituted for the distance
    encoder at the fixed weight, on the frozen 444 pairs (primary), the
    legacy 905 and their union, all from expanded-population models."""
    if not CONFIRM.exists():
        return
    c = json.loads(CONFIRM.read_text())
    rec = c["sources"].get("attn")
    if not rec or "fresh" not in rec:
        return
    pops = [("fresh", "frozen 444\n(held out, one look)"),
            ("legacy", "legacy 905"), ("all", "union 1,349")]
    systems = [("tabular", "tabular level/shape\n(CatBoost)", "#9a9a9a"),
               ("blend_dist", "+ distance encoder\nshape, w 0.35", "#4a4a4a"),
               ("blend_src", "+ attention encoder\nshape, w 0.35", "#1f77b4")]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.16)
    xs = np.arange(len(pops)); wd = 0.26
    for j, (key, lab, col) in enumerate(systems):
        vals = [rec[p][key]["r2"] for p, _ in pops]
        bars = ax.bar(xs + (j - 1) * wd, vals, wd, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:+.4f}",
                    ha="center", va="bottom", fontsize=8.3)
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in pops], fontsize=9.5)
    ax.set_ylabel("adjacent-pair log SF R$^2$")
    ax.set_ylim(0, max(rec[p]["blend_dist"]["r2"] for p, _ in pops) * 1.45)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=3)
    f = rec["fresh"]
    verdict = ("PASS" if rec["primary_pass"] else "FAIL")
    ax.set_title(f"Held-out confirmation under the frozen rule: attention vs tabular "
                 f"{f['primary_contrast']:+.4f} on the 444 ({verdict})\n"
                 f"attention vs distance encoder: {f['vs_dist_contrast']:+.4f} on the 444, "
                 f"{rec['legacy']['vs_dist_contrast']:+.4f} on the 905, "
                 f"{rec['all']['vs_dist_contrast']:+.4f} on the union",
                 fontsize=9.6)
    fig.savefig(FIG2, dpi=170)
    print(f"wrote {FIG2}")


if __name__ == "__main__":
    raise SystemExit(main())
