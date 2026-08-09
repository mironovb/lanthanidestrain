"""Per-lanthanide aqueous-phase and f-shell constants.

Why the model needs these
-------------------------
Everything per-metal that currently reaches any model is ``Atomic Number_metal``,
``lanthanide_index``, ``Ionic Radius_metal`` and ``metal_ox`` -- four columns,
three of them monotone in the same thing, and the fourth constant (every row is
Ln(III)).  Solvent-extraction selectivity is

    Delta(complexation)  -  Delta(DEHYDRATION)

and there is no aqueous-phase quantity in the feature set at all.  A model whose
only series coordinate is the ionic radius can express a monotone trend and
nothing else.

The data says that is not enough.  Reconstructing the metric's own pairs, mean
``dy`` by pair index is non-monotone and changes sign twice across the series,
and a leave-extractants-out baseline predicting ``dy`` from PAIR IDENTITY ALONE
already reaches R2 ~ 0.058 -- a reproducible, ligand-independent series shape
that nothing in the current featurisation expresses explicitly.

Why this is not sweep2's A1 collapse again
------------------------------------------
A1 added 119 geometry columns and cost -0.3167, because they were
arbitrary-conformer artefacts: median |corr(d feature, dy)| = 0.0495 against the
``cond`` block's 0.0804.  Every column here is a LOOKUP CONSTANT with zero
conformer scatter by construction -- the same property that makes the incumbent
``Ionic Radius_metal`` work.  That is an argument, not a result, so every column
is put through ``automl.topo.within_block_signal`` before it is trained on, and
anything that does not clear the bar does not get a GPU run.

Sources
-------
Ionic radii (CN 8, Shannon 1976) are already in the dataset and are NOT
duplicated here.  Hydration enthalpies and free energies follow the standard
Marcus compilation; aqua-ion coordination numbers follow the well-documented
9 -> 8 change across the series (the "gadolinium break"); third ionisation
energies are NIST atomic-spectra values.  Values are given to the precision at
which they are quoted; they enter standardised per fold, so absolute scale is
irrelevant and only the SHAPE across the series carries information.

Pm (index 5) is included for completeness and never occurs in the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# symbol: (Z, n_f electrons in Ln(III), aqua CN, -dH_hyd kJ/mol,
#          -dG_hyd kJ/mol, IE3 eV, Ln-OH2 distance A)
LN_PHYSICS: dict[str, tuple[int, int, float, float, float, float, float]] = {
    "La": (57,  0, 9.0, 3283.0, 3145.0, 19.177, 2.58),
    "Ce": (58,  1, 9.0, 3326.0, 3193.0, 20.198, 2.55),
    "Pr": (59,  2, 9.0, 3373.0, 3239.0, 21.624, 2.54),
    "Nd": (60,  3, 9.0, 3403.0, 3271.0, 22.100, 2.51),
    "Pm": (61,  4, 9.0, 3427.0, 3300.0, 22.300, 2.50),
    "Sm": (62,  5, 9.0, 3449.0, 3329.0, 23.424, 2.48),
    "Eu": (63,  6, 9.0, 3501.0, 3360.0, 24.920, 2.47),
    "Gd": (64,  7, 8.5, 3517.0, 3383.0, 20.630, 2.45),
    "Tb": (65,  8, 8.2, 3559.0, 3417.0, 21.910, 2.43),
    "Dy": (66,  9, 8.0, 3567.0, 3438.0, 22.800, 2.41),
    "Ho": (67, 10, 8.0, 3623.0, 3465.0, 22.840, 2.40),
    "Er": (68, 11, 8.0, 3637.0, 3490.0, 22.740, 2.38),
    "Tm": (69, 12, 8.0, 3664.0, 3514.0, 23.680, 2.37),
    "Yb": (70, 13, 8.0, 3706.0, 3542.0, 25.050, 2.35),
    "Lu": (71, 14, 8.0, 3739.0, 3564.0, 25.426, 2.34),
}

PREFIX = "mphys__"


def metal_physics_frame() -> pd.DataFrame:
    """One row per lanthanide symbol, indexed by symbol."""
    rows = {}
    for sym, (_z, nf, cn, dh, dg, ie3, dw) in LN_PHYSICS.items():
        f = float(nf)
        rows[sym] = {
            f"{PREFIX}n_f": f,
            f"{PREFIX}dH_hyd": dh,
            f"{PREFIX}dG_hyd": dg,
            f"{PREFIX}aqua_cn": cn,
            f"{PREFIX}ie3": ie3,
            f"{PREFIX}d_LnOH2": dw,
            # --- tetrad coordinates -------------------------------------
            # The nephelauxetic / tetrad effect divides the series at the
            # quarter, half and three-quarter f-shell.  These give the model a
            # coordinate that is NOT monotone in radius, which is the whole
            # point: a monotone coordinate cannot bend where mean dy bends.
            f"{PREFIX}tetrad_q1": abs(f - 3.5),
            f"{PREFIX}tetrad_q2": abs(f - 7.0),
            f"{PREFIX}tetrad_q3": abs(f - 10.5),
            # Half-shell / filled-shell stability, the Eu(II) and Yb(II)
            # anomalies that make IE3 non-monotone.
            f"{PREFIX}half_shell": float(nf in (6, 7, 13, 14)),
            # Hydration energy PER unit charge density is the quantity that
            # competes with complexation; radius is joined in below.
            f"{PREFIX}cn_break": float(cn < 9.0),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def attach(df: pd.DataFrame, metal_col: str = "metal",
           radius_col: str = "Ionic Radius_metal") -> list[str]:
    """Add the block to ``df`` in place; return the column names.

    Unknown symbols get NaN rather than a guess -- ``_standardise`` imputes the
    training-fold median, which is the honest behaviour for a metal whose
    constants are not tabulated, and silently inventing one would be worse.
    """
    tab = metal_physics_frame()
    sym = df[metal_col].astype(str)
    for c in tab.columns:
        df[c] = sym.map(tab[c]).astype(float)
    # Two ratios that need a dataset column, so they cannot live in the table.
    if radius_col in df.columns:
        r = pd.to_numeric(df[radius_col], errors="coerce")
        # Charge density: the Born term driving dehydration cost.
        df[f"{PREFIX}z_over_r"] = 3.0 / r
        # Hydration free energy against the Born prediction.  The RESIDUAL is
        # the part a radius-only model cannot see, and it is where the
        # tetrad/CN-break structure lives.
        born = df[f"{PREFIX}dG_hyd"] * r
        df[f"{PREFIX}dG_born_resid"] = (df[f"{PREFIX}dG_hyd"]
                                        - born / r.mean()) / 1.0
    return [c for c in df.columns if c.startswith(PREFIX)]


def series_shape() -> pd.Series:
    """Sanity display: is any column non-monotone across the series?"""
    tab = metal_physics_frame()
    order = [s for s in LN_PHYSICS]
    out = {}
    for c in tab.columns:
        v = tab.loc[order, c].to_numpy(float)
        d = np.diff(v)
        out[c] = float(np.mean(np.sign(d) != np.sign(d[np.argmax(np.abs(d))])))
    return pd.Series(out, name="frac_steps_against_dominant_direction")
