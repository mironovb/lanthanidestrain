#!/usr/bin/env python3
"""Do the swept configurations actually change the representation?

Why this exists
---------------
The likely outcome of this sweep is that every configuration ties at stack weight
0.00, because the published persistence-image arm already does. If that happens,
there are two completely different explanations and the training runs alone
cannot tell them apart:

**(a) The images barely changed.** Resolution and spread mostly re-express the
same information, so no readout could distinguish the configurations and a tie
says nothing about persistence homology.

**(b) The images changed substantially and it did not help.** Then the tie is
informative: the representation moved and the target still could not use it.

This measures which, from the rendered images alone -- no training, no ``log D``,
and therefore nothing that could bias the endpoint.

How
---
Pixel-wise comparison is impossible across resolutions, so the comparison is on
the **geometry each configuration induces over the dataset**: the 953 x 953
matrix of pairwise distances between complexes in image space. That is
resolution-independent -- it asks whether two complexes that look similar under
the shipped settings still look similar under a swept one.

Two numbers per configuration, both against the shipped anchor:

``rho``
    Spearman correlation between the two configurations' pairwise-distance
    vectors (a Mantel-style statistic). 1.0 means the configuration re-expresses
    exactly the same relationships between complexes; lower means it genuinely
    reorders them.

``eff_dim``
    Participation ratio of the image set's covariance eigenvalues,
    ``(sum L)^2 / sum(L^2)``. How many directions the representation actually
    varies in -- a 128x128 image with effective dimension 3 carries no more
    usable structure than a 20x20 one.

Images are compared after the same ``log1p`` and per-channel standardisation
``PersistenceImages`` applies, so this describes what the network is handed
rather than the raw render.

``data/`` is never written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "automl/artifacts/pi_sweep"
REPORTS = REPO / "automl/reports"


def prepared(path: Path) -> np.ndarray:
    """Images as the network receives them: log1p, per-channel standardised."""
    with np.load(path) as z:
        im = z["images"].astype(np.float32)
    im = np.log1p(im)
    s = im.std(axis=(0, 2, 3), keepdims=True)
    return (im / np.where(s > 1e-8, s, 1.0)).reshape(len(im), -1)


def pairwise(X: np.ndarray) -> np.ndarray:
    """Condensed pairwise Euclidean distances, via the Gram matrix."""
    sq = (X ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(len(X), k=1)
    return np.sqrt(d2[iu])


def eff_dim(X: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # Eigenvalues of the (n x n) Gram matrix equal those of the covariance, and
    # n = 953 is far smaller than the 16,384 pixels at the top resolution.
    g = Xc @ Xc.T
    lam = np.linalg.eigvalsh(g)
    lam = lam[lam > 1e-10]
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if len(lam) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("a", "b"), default="a")
    args = ap.parse_args()

    from scipy.stats import spearmanr

    man = SWEEP / f"manifest_stage_{args.stage}.json"
    rows_cfg = json.loads(man.read_text())
    anchor = next((r for r in rows_cfg
                   if r["resolution"] == 20 and abs(r["spread"] - 0.08) < 1e-9
                   and r["channels"] == "sum" and r["weight"] == "linear"
                   and abs(r["hi"] - 2.5) < 1e-9), None)
    if anchor is None:
        raise SystemExit("no shipped anchor in this manifest")

    Xa = prepared(Path(anchor["path"]))
    da = pairwise(Xa)
    print(f"anchor: {anchor['resolution']}px spread {anchor['spread']:.4f}  "
          f"effective dimension {eff_dim(Xa):.1f}\n")

    out = []
    for r in rows_cfg:
        X = prepared(Path(r["path"]))
        rho = 1.0 if r["key"] == anchor["key"] else float(
            spearmanr(da, pairwise(X)).statistic)
        ed = eff_dim(X)
        px = r["spread"] * (r["resolution"] - 1) / (r["hi"] - r["lo"])
        out.append({"key": r["key"], "resolution": r["resolution"],
                    "spread": r["spread"], "spread_px": px,
                    "hi": r["hi"], "weight": r["weight"],
                    "channels": r["channels"], "n_pixels": X.shape[1],
                    "rho_vs_anchor": rho, "eff_dim": ed})
        print(f"  {r['resolution']:4d}px {px:4.1f}px spread  {r['channels']:5s} "
              f"{r['weight']:8s} hi={r['hi']:.1f}   rho={rho:.4f}  "
              f"eff_dim={ed:5.1f}")

    df = pd.DataFrame(out)
    csv = REPORTS / f"pi_sweep_geometry_{args.stage}.csv"
    df.to_csv(csv, index=False)

    swept = df[df["key"] != anchor["key"]]
    print(f"\nacross {len(swept)} swept configurations:")
    print(f"  rho vs anchor: min {swept['rho_vs_anchor'].min():.4f}  "
          f"median {swept['rho_vs_anchor'].median():.4f}  "
          f"max {swept['rho_vs_anchor'].max():.4f}")
    print(f"  effective dimension: {df['eff_dim'].min():.1f} - "
          f"{df['eff_dim'].max():.1f} "
          f"(pixels range {df['n_pixels'].min()} - {df['n_pixels'].max()})")

    lo = swept["rho_vs_anchor"].min()
    if lo > 0.99:
        print("\n=> The configurations are near-identical in the geometry they")
        print("   induce over the dataset. A tie in the training results would")
        print("   then say little about persistence homology: no readout could")
        print("   separate representations this similar.")
    else:
        print(f"\n=> The configurations genuinely differ (rho down to {lo:.3f}).")
        print("   A tie in the training results would therefore be informative:")
        print("   the representation moved and the target still could not use it.")
    print(f"  -> {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
