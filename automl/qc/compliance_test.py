#!/usr/bin/env python3
"""Does the computed per-ligand compliance predict measured selectivity?

The question, and why it is the only one left
---------------------------------------------
g-xTB fixes the chemistry: against Shannon (1976) radii the computed M-donor
contraction has slope **1.142** (exact = 1.00) where GFN2 gives **0.386** and
scatters over a six-fold range across ligands.  It also carries real f-shell
structure that GFN2 cannot represent -- 370x more departure from linear-in-Z,
reproducing at r = +0.97.

None of that is automatically worth anything here, because **96.1 %** of that
non-linear response is *shared across ligands* -- a pure function of metal
identity.  Metal identity is already recoverable from the token at R2 = 0.9995,
so a metal-only function is information the model has had all along.

What is left is the ligand-specific coefficient

    c_L = d<M-donor> / d r_Shannon      (one number per ligand)

and this is not a bookkeeping detail: adjacent-lanthanide selectivity *is*
ligand-dependent discrimination.  If Delta d ~ c_L * Delta r and c_L were
universal, every ligand would discriminate identically and geometry would
predict **zero** selectivity differences.  On six ligands c_L had cv = 0.059,
which is close enough to universal to be alarming -- but six ligands cannot
separate physics from numerics, and all six were O/N donors at CN 8-9.

This module tests c_L on ~70 ligands against the measured separations.

The prediction, stated before the numbers are read
--------------------------------------------------
If the structural picture holds, a ligand whose donor shell responds *more* to
the ionic radius should discriminate adjacent lanthanides *more*, so

    corr( c_L , per-ligand adjacent |log SF| )  >  0.

Two controls decide whether a positive result means anything:

1. **GFN2 arm.**  Its c_L is known to be mostly noise (23 % shared across
   ligands, vs g-xTB's 96 %).  If GFN2's c_L predicts selectivity *as well as*
   g-xTB's, the correlation is not coming from the physics.
2. **Ligand size.**  c_L, atom count and measured selectivity are all
   confounded with how big and how chelating the ligand is.  The partial
   correlation controlling for atom count and denticity is the honest number.

Measured selectivity uses ``evaluation.adjacent_pair_arrays`` -- the *same*
construction as the metric being modelled (group on composition_key, average
replicates per (block, metal), difference |delta index| == 1).  Re-deriving it
here is how a figure once reported 13,029 pairs and inverted the real result.

    python3 -m automl.qc.compliance_test --tags cf_shard0,cf_shard1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.evaluation import adjacent_pair_arrays  # noqa: E402

SERIES = _REPO / "automl/artifacts/gxtb_series"
DATASET = _REPO / "data/processed/final_ml_dataset_3d.parquet"

# Shannon (1976) effective ionic radii for Ln(III), angstrom.
SHANNON = {
    8: {"La": 1.160, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109, "Pm": 1.093,
        "Sm": 1.079, "Eu": 1.066, "Gd": 1.053, "Tb": 1.040, "Dy": 1.027,
        "Ho": 1.015, "Er": 1.004, "Tm": 0.994, "Yb": 0.985, "Lu": 0.977},
    9: {"La": 1.216, "Ce": 1.196, "Pr": 1.179, "Nd": 1.163, "Pm": 1.144,
        "Sm": 1.132, "Eu": 1.120, "Gd": 1.107, "Tb": 1.095, "Dy": 1.083,
        "Ho": 1.072, "Er": 1.062, "Tm": 1.052, "Yb": 1.042, "Lu": 1.032},
}


def load_series(tags: list[str]) -> list[dict]:
    recs = []
    for t in tags:
        p = SERIES / f"{t}.json"
        if not p.exists():
            print(f"  [skip] {p.name} not present")
            continue
        d = json.loads(p.read_text())
        recs.extend([r for r in d["records"] if r.get("ok")])
        print(f"  [load] {p.name}: {len(d['records'])} records")
    return recs


def compliance(recs: list[dict]) -> pd.DataFrame:
    """One c_L per (ligand, arm): slope of <M-donor> on the Shannon radius.

    La is dropped: it is a separate GFN2 parameter anchor (off-trend by 15x),
    and including it would let one arm's known parameter discontinuity masquerade
    as ligand chemistry.
    """
    rows = []
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame()
    for (fam, arm), g in df.groupby(["family", "arm"]):
        g = g[g.f_count >= 1]
        if len(g) < 8:
            continue
        cn = int(g.iloc[0].get("cn") or 9)
        sh = SHANNON.get(cn, SHANNON[9])
        x = g.metal.map(sh).to_numpy(dtype=float)
        y = g.mean_m_donor.to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            continue
        c, b = np.polyfit(x[ok], y[ok], 1)
        pred = c * x[ok] + b
        ss = ((y[ok] - y[ok].mean()) ** 2).sum()
        rows.append({"ligand": fam.split("||")[0], "family": fam, "arm": arm,
                     "c_L": float(c), "n_metals": int(ok.sum()),
                     "cn": cn, "n_atoms": int(g.iloc[0].get("n_atoms") or 0),
                     "fit_r2": float(1 - ((y[ok] - pred) ** 2).sum() / ss)
                                if ss > 0 else np.nan,
                     "resid_sd_ang": float(np.std(y[ok] - pred, ddof=1))})
    return pd.DataFrame(rows)


def measured_selectivity() -> pd.DataFrame:
    """Per-ligand adjacent-lanthanide discrimination, from the data.

    ``adjacent_pair_arrays`` is called with the prediction slot filled by the
    truth, so the returned "predicted" differences are ignored; only the true
    separations are used.  Reusing the function rather than reimplementing it
    keeps this consistent with ``sel_adj_logSF_r2`` by construction.
    """
    # ``composition_key`` is DERIVED (dataset.py builds it from the group column
    # plus binned conditions), not a stored column.  Reconstructing it here
    # would be the c6_split mistake again -- that drew on 187 extractants from
    # the matrix cache when only 162 were modelled.  Go through the same loader
    # the trainer uses.
    from automl.topo.train import build_row_table  # noqa: PLC0415
    df, _, _ = build_row_table(preset="baseline_2d", arch="dist",
                               match_rows="dist")
    need = ["LIGAND_SMILES", "composition_key", "lanthanide_index", "log_D"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"row table lacks {missing}; columns look like "
                         f"{[c for c in df.columns if 'SMIL' in c.upper()][:5]}")
    df = df[need].dropna()
    out = []
    for lig, g in df.groupby("LIGAND_SMILES"):
        y = g.log_D.to_numpy(dtype=float)
        dy, _ = adjacent_pair_arrays(y, y, g.composition_key.to_numpy(),
                                     g.lanthanide_index.to_numpy())
        if len(dy) < 3:
            continue
        out.append({"ligand": lig, "n_adj_pairs": len(dy),
                    "mean_abs_logSF": float(np.mean(np.abs(dy))),
                    "sd_logSF": float(np.std(dy, ddof=1)),
                    "p90_abs_logSF": float(np.percentile(np.abs(dy), 90))})
    return pd.DataFrame(out)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def _partial(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """Correlation of x and y after regressing both on the confounders Z."""
    Z = np.column_stack([Z, np.ones(len(x))])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", default="cf_shard0,cf_shard1")
    ap.add_argument("--min-pairs", type=int, default=5)
    args = ap.parse_args()

    print("[compliance_test] loading series")
    recs = load_series([t for t in args.tags.split(",") if t])
    if not recs:
        raise SystemExit("no records; are the runs finished?")
    C = compliance(recs)
    if C.empty:
        raise SystemExit("no compliance fits")
    print(f"\ncompliance fits: {len(C)} (ligand x arm), "
          f"{C.ligand.nunique()} distinct ligands")
    for arm, g in C.groupby("arm"):
        print(f"  {arm:9s} n={len(g):3d}  c_L mean={g.c_L.mean():.3f} "
              f"sd={g.c_L.std():.4f}  cv={g.c_L.std()/g.c_L.mean():.3f}  "
              f"median fit R2={g.fit_r2.median():.3f}")

    S = measured_selectivity()
    print(f"\nmeasured selectivity: {len(S)} ligands with >=3 adjacent pairs")
    S = S[S.n_adj_pairs >= args.min_pairs]

    print(f"\n{'='*72}\nDOES c_L PREDICT MEASURED ADJACENT-LANTHANIDE SELECTIVITY?\n{'='*72}")
    print(f"{'arm':9s} {'n':>4s} {'pearson':>9s} {'spearman':>9s} "
          f"{'partial(size,cn)':>17s}")
    for arm, g in C.groupby("arm"):
        m = g.merge(S, on="ligand", how="inner")
        if len(m) < 8:
            print(f"{arm:9s} {len(m):4d}   too few matched ligands")
            continue
        x = m.c_L.to_numpy(); y = m.mean_abs_logSF.to_numpy()
        Z = np.column_stack([m.n_atoms.to_numpy(float), m.cn.to_numpy(float)])
        print(f"{arm:9s} {len(m):4d} {np.corrcoef(x, y)[0,1]:+9.4f} "
              f"{_spearman(x, y):+9.4f} {_partial(x, y, Z):+17.4f}")

    out = SERIES / "compliance_test.json"
    payload = {"compliance": C.to_dict("records"),
               "selectivity": S.to_dict("records")}
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
