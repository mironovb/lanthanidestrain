#!/usr/bin/env python3
"""Read-out of the An/Ln transfer experiments (one look, pre-frozen pairs).

Inputs: automl/reports/an_ln/{ln_only,ln_am}.json (per-seed and ensemble
scores) and automl/artifacts/an_ln/oof_{ln_only,ln_am}_ens4.parquet.
Outputs: automl/reports/an_ln/summary.json and am_eu_transfer.png.

  * checks that the scored Am/Eu pairs are exactly the frozen ones;
  * paired-seed contrasts ln_am - ln_only on the legacy Ln metric and on
    Am/Eu (same seeds, same folds);
  * per-ligand-family Am/Eu quality of the joint model (families by SMARTS:
    DGA / BTP-BTBP-BTPhen / CMPO-carbamoylphosphine / phosphine oxide /
    malonamide / other);
  * scatter of measured vs predicted Am/Eu log SF, zero-shot vs joint.

Usage:  PYTHONPATH=$PWD python3 -m automl.an_ln.report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats

import automl.evaluation as ev

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/an_ln"
REP = REPO / "automl/reports/an_ln"

FAMILY_SMARTS = [
    ("DGA", "O=C(N)COCC(=O)N"),
    ("BTP/BTBP/BTPhen", "c1nnc(nc1)-c1ncccc1"),
    ("CMPO", "O=C(N)CP(=O)"),
    ("malonamide", "O=C(N)CC(=O)N"),
    ("aza-aromatic (thio)amide", "[CX3](=[O,S])(N)c1ncccc1"),
    ("azolyl-pyridine (N-DPP, PyTri)", "[$(n1nccc1-c1ncccc1),$(c1cn(nn1)),$(c1nn(cc1)-c1ncccc1)]"),
    ("thiophosphoryl (S donors)", "P(=S)"),
    ("phosphine oxide / phosphate", "P(=O)"),
    ("other amide", "C(=O)N"),
]
_PATS = [(n, Chem.MolFromSmarts(s)) for n, s in FAMILY_SMARTS]


def family(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "other"
    for name, pat in _PATS:
        if m.HasSubstructMatch(pat):
            return name
    return "other"


def am_eu_pairs(oof: pd.DataFrame, an="Am", ln="Eu") -> pd.DataFrame:
    cells = oof.groupby(["composition_key", "metal"], as_index=False).agg(
        y=("y", "mean"), p=("oof", "mean"), ex=("extractant_group", "first"))
    rows = []
    for ck, blk in cells.groupby("composition_key"):
        b = blk.set_index("metal")
        if an in b.index and ln in b.index:
            dy = b.loc[an, "y"] - b.loc[ln, "y"]
            if dy == 0.0:
                continue
            rows.append({"ck": ck, "ex": b.loc[an, "ex"], "dy": dy,
                         "dp": b.loc[an, "p"] - b.loc[ln, "p"]})
    return pd.DataFrame(rows)


def score(P: pd.DataFrame) -> dict:
    return {"n": int(len(P)), "r2": round(ev._r2(P.dy.to_numpy(), P.dp.to_numpy()), 4),
            "spearman": round(float(stats.spearmanr(P.dy, P.dp).statistic), 3),
            "sign_acc": round(float(np.mean(np.sign(P.dy) == np.sign(P.dp))), 3),
            "sd_dy": round(float(P.dy.std()), 3), "sd_dp": round(float(P.dp.std()), 3)}


def main() -> int:
    J = {c: json.loads((REP / f"{c}.json").read_text()) for c in ("ln_only", "ln_am")}
    O = {c: pd.read_parquet(ART / f"oof_{c}_ens4.parquet") for c in J}
    frozen = json.loads((ART / "frozen_pairs.json").read_text())
    P = {c: am_eu_pairs(O[c]) for c in J}
    assert set(P["ln_am"].ck) == set(P["ln_only"].ck) == set(frozen["Am_Eu"]["block_keys"]), \
        "scored Am/Eu pairs differ from the frozen set"
    out = {"frozen_check": f"{len(P['ln_am'])} Am/Eu pairs == frozen set", "ensemble": {}}
    for c in J:
        out["ensemble"][c] = {"ln_legacy_r2": round(J[c]["ensemble"]["ln_legacy"]["r2"], 4),
                              "am_eu": score(P[c]),
                              "cm_eu": J[c]["ensemble"]["cm_eu"],
                              "am_cm": J[c]["ensemble"]["am_cm"]}
    # paired seeds
    s0 = {r["seed"]: r for r in J["ln_only"]["per_seed"]}
    s1 = {r["seed"]: r for r in J["ln_am"]["per_seed"]}
    seeds = sorted(set(s0) & set(s1))
    d_ln = [s1[s]["ln_legacy"]["r2"] - s0[s]["ln_legacy"]["r2"] for s in seeds]
    d_am = [s1[s]["am_eu"]["r2"] - s0[s]["am_eu"]["r2"] for s in seeds]
    out["paired_seeds"] = {
        "seeds": seeds,
        "ln_legacy_ln_only": [round(s0[s]["ln_legacy"]["r2"], 4) for s in seeds],
        "ln_legacy_ln_am": [round(s1[s]["ln_legacy"]["r2"], 4) for s in seeds],
        "d_ln_legacy_mean": round(float(np.mean(d_ln)), 4),
        "d_ln_legacy_sd": round(float(np.std(d_ln, ddof=1)), 4),
        "d_ln_legacy_all_positive": bool(all(x > 0 for x in d_ln)),
        "am_eu_ln_only": [round(s0[s]["am_eu"]["r2"], 4) for s in seeds],
        "am_eu_ln_am": [round(s1[s]["am_eu"]["r2"], 4) for s in seeds],
        "d_am_eu_mean": round(float(np.mean(d_am)), 4),
    }
    # families (joint model)
    Pj = P["ln_am"].copy(); Pj["family"] = Pj.ex.map(family)
    fam = {}
    for f, g in Pj.groupby("family"):
        if len(g) >= 8:
            fam[f] = dict(score(g), n_extractants=int(g.ex.nunique()),
                          mean_abs_dy=round(float(g.dy.abs().mean()), 3))
    out["am_eu_by_family_joint"] = fam
    # also: does the joint model rank extractants for Am/Eu?  (one value per extractant)
    e = Pj.groupby("ex")[["dy", "dp"]].mean()
    out["am_eu_per_extractant"] = {"n": int(len(e)),
                                   "spearman": round(float(stats.spearmanr(e.dy, e.dp).statistic), 3),
                                   "r2": round(ev._r2(e.dy.to_numpy(), e.dp.to_numpy()), 4)}
    # control: how much of the pooled score is family membership alone?
    # leave-one-extractant-out family mean of measured dy (no model at all),
    # and the same for the model's own predictions pooled within family.
    fm = []
    for i, r in Pj.iterrows():
        others = Pj[(Pj.family == r.family) & (Pj.ex != r.ex)]
        fm.append(others.dy.mean() if len(others) else Pj[Pj.ex != r.ex].dy.mean())
    Pj["family_mean"] = fm
    out["control_family_mean_loeo"] = {
        "r2": round(ev._r2(Pj.dy.to_numpy(), Pj.family_mean.to_numpy()), 4),
        "spearman": round(float(stats.spearmanr(Pj.dy, Pj.family_mean).statistic), 3),
        "note": "predict a held-out extractant's Am/Eu by the mean of the other "
                "extractants in its SMARTS family; no model"}
    # joint model, family-demeaned (within-family skill only)
    dm_y = Pj.dy - Pj.groupby("family").dy.transform("mean")
    dm_p = Pj.dp - Pj.groupby("family").dp.transform("mean")
    out["joint_within_family"] = {
        "r2": round(ev._r2(dm_y.to_numpy(), dm_p.to_numpy()), 4),
        "spearman": round(float(stats.spearmanr(dm_y, dm_p).statistic), 3),
        "note": "both measured and predicted Am/Eu centred on their family mean"}
    (REP / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    # figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    for ax, c, title in zip(axes, ("ln_only", "ln_am"),
                            ("zero-shot: Ln-trained, Am as a pseudo-lanthanide",
                             "joint training: Ln + Am rows, 5f flag")):
        Q = P[c]; sc = score(Q)
        ax.scatter(Q.dy, Q.dp, s=14, alpha=0.6, color="#2a78d6" if c == "ln_am" else "#9aa1a8")
        lim = [-2.6, 3.4]
        ax.plot(lim, lim, ls=":", color="#52514e", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{title}\nR² = {sc['r2']:+.3f}, Spearman ρ = {sc['spearman']:+.2f}, "
                     f"n = {sc['n']} pairs / {Q.ex.nunique()} extractants", fontsize=10)
        ax.set_xlabel("measured log SF(Am/Eu)")
    axes[0].set_ylabel("predicted log SF(Am/Eu)  (extractants held out)")
    fig.tight_layout(); fig.savefig(REP / "am_eu_transfer.png", dpi=200)
    print("wrote", REP / "am_eu_transfer.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
