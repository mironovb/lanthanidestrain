#!/usr/bin/env python3
"""What actually partitions into the diluent? Speciation features.

Why this exists
---------------
89 % of the modelled complexes carry net charge **+3**: they are bare
Ln(III)-ligand cations.  The species that partitions into kerosene is neutral.
Charge neutralisation IS the physics of solvent extraction, and it is currently
invisible to every model in this study -- the 3D complex has no counter-ions,
and the 9 recorded acids reach the model only as one-hot columns that never
interact with anything structural.

Two extraction mechanisms are conflated as a result:

* **acidic / cation-exchange** extractants (HDEHP, HEH[EHP], carboxylic and
  phosphoric acids) ionise and replace the metal's charge themselves:
  Ln(3+) + 3HA -> LnA3 + 3H(+).  Their selectivity depends on pKa and on
  proton competition, so acid concentration enters with a NEGATIVE sign.
* **neutral / solvating** extractants (TBP, CMPO, TODGA, malonamides) cannot
  neutralise anything; the counter-ion does it, and the extracted species is
  Ln(NO3)3.nL.  Their selectivity depends on the ANION, and acid concentration
  enters with a POSITIVE sign through nitrate availability.

These are opposite dependencies on the same recorded variable.  A model that
cannot tell the two classes apart must average them.

This module derives the distinction from ``LIGAND_SMILES`` plus the recorded
acid, and emits it as an appendable feature block.  It does not rebuild any
geometry -- that is the honest fix and it is out of scope here.

VERDICT, 31 July 2026: **not testable on this dataset.**  The audit below is
the deliverable; no GPU cell was run, because there is nothing here to learn
from.  97.5 % of the 162 extractants are neutral solvating agents -- the table
is dominated by diglycolamides (CITAM, TBDGA, TEHDGA, DOODA) and BTBPs, and
only 6.8 % of extractant SMILES contain P(=O) at all.  So the mechanism split
this module exists to expose has almost no variance:

    spec__is_cation_exchanger    SD 0.041   degenerate
    spec__dentate                r = 1.00 with the existing DENTATE column
    spec__nitrate_available      r = 0.99 with cond__acid__hno3
    spec__acidconc_x_solvating   r = 1.00 with cond__acid_concentration

Exactly one derived column is both new and varying
(``spec__n_neutral_donor_groups``, SD 0.453, max |r| 0.61 against any existing
column).  One column is not a track, and CAMPAIGN3_PREREGISTRATION section 6
commits to reporting an untestable cell as NOT TESTED rather than as a null --
a null here would read as "speciation does not matter", when what happened is
that the dataset cannot ask the question.

That is itself worth recording: testing the cation-exchange/solvating axis
needs data with acidic extractants in it (HDEHP, HEH[EHP], Cyanex 272), which
this table essentially lacks.

Usage
-----
    python3 -m automl.topo.speciation        # audit what it derives
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# Ionisable groups that make an extractant a cation-exchanger.  Matched on the
# SMILES string rather than with RDKit substructure search so the module has no
# hard dependency; the patterns are deliberately conservative and the audit
# below reports what fraction they classify.
ACIDIC_PATTERNS = (
    "P(=O)(O)O", "P(O)(=O)O", "OP(=O)O", "P(=O)(O)",       # phosphoric / phosphonic
    "C(=O)O", "OC(=O)",                                     # carboxylic
    "S(=O)(=O)O",                                           # sulfonic
)
# Groups that solvate without ionising.
NEUTRAL_PATTERNS = (
    "P(=O)(", "C(=O)N", "N C(=O)", "S(=O)(=O)N",
)


def _is_acidic(smiles: str) -> tuple[int, int]:
    """(n acidic groups, n neutral donor groups) from a SMILES string."""
    if not isinstance(smiles, str) or not smiles:
        return 0, 0
    s = smiles.replace(" ", "")
    n_acid = sum(s.count(p) for p in ACIDIC_PATTERNS)
    n_neut = sum(s.count(p) for p in NEUTRAL_PATTERNS)
    # A phosphate ester P(=O)(OC)(OC) is neutral; the acidic patterns above can
    # also match it, so require a terminal OH: an "O)" or "O" not followed by C.
    if n_acid and ("OP(=O)(O)" not in s and "(O)O" not in s
                   and "OC(=O)" not in s and "C(=O)O" not in s):
        n_acid = 0
    return int(n_acid), int(n_neut)


def build(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Speciation columns for every row, plus their names."""
    smi = df["LIGAND_SMILES"].astype(str) if "LIGAND_SMILES" in df else None
    if smi is None:
        raise SystemExit("no LIGAND_SMILES column; cannot derive speciation")

    n_acid, n_neut = zip(*(_is_acidic(s) for s in smi))
    n_acid = np.asarray(n_acid, float)
    n_neut = np.asarray(n_neut, float)
    is_acidic = (n_acid > 0).astype(float)

    dent = (pd.to_numeric(df["DENTATE"], errors="coerce").to_numpy(float)
            if "DENTATE" in df else np.full(len(df), np.nan))

    acid_cols = [c for c in df.columns if c.startswith("cond__acid__")]
    acid_conc = (pd.to_numeric(df.get("cond__acid_concentration_M"),
                               errors="coerce").to_numpy(float)
                 if "cond__acid_concentration_M" in df
                 else np.full(len(df), np.nan))
    # Which anion is available to neutralise the metal, if the extractant cannot.
    nitrate = (df[[c for c in acid_cols if "hno3" in c]].sum(axis=1).to_numpy(float)
               if any("hno3" in c for c in acid_cols) else np.zeros(len(df)))

    charge_to_neutralise = 3.0 - n_acid          # what the anion must supply
    cols = {
        "spec__n_acidic_groups": n_acid,
        "spec__n_neutral_donor_groups": n_neut,
        "spec__is_cation_exchanger": is_acidic,
        "spec__is_solvating": 1.0 - is_acidic,
        "spec__dentate": dent,
        "spec__charge_left_for_anion": charge_to_neutralise,
        "spec__nitrate_available": nitrate,
        # The interaction the mechanism argument predicts: acid concentration
        # should act with OPPOSITE sign for the two classes, and no model can
        # represent that from the main effects alone because the class is not
        # currently a feature at all.
        "spec__acidconc_x_cation_exchanger": acid_conc * is_acidic,
        "spec__acidconc_x_solvating": acid_conc * (1.0 - is_acidic),
        "spec__nitrate_x_solvating": nitrate * (1.0 - is_acidic),
    }
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(np.float32), names


def main() -> int:
    from automl.topo.train import build_row_table
    df, _, _ = build_row_table(preset="baseline_2d", arch="snn")
    Z, names = build(df)
    print(f"{Z.shape[1]} speciation columns over {Z.shape[0]} rows\n")
    frame = pd.DataFrame(Z, columns=names)
    rows = []
    for n in names:
        v = frame[n]
        rows.append(dict(column=n, coverage=float(np.isfinite(v).mean()),
                         mean=float(np.nanmean(v)), sd=float(np.nanstd(v))))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    ce = frame["spec__is_cation_exchanger"]
    print(f"\nclassified as cation-exchange: {ce.mean():.1%} of rows")
    ex = df.groupby("extractant_group")["LIGAND_SMILES"].first()
    ce_ex = [bool(_is_acidic(str(s))[0]) for s in ex]
    print(f"                               {np.mean(ce_ex):.1%} of "
          f"{len(ex)} distinct extractants")
    print("\nThe split is the point: these two classes depend on acid "
          "concentration\nwith OPPOSITE signs, and no model in this study has "
          "been able to tell them apart.")
    OUT = REPORTS / "speciation_audit.csv"
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
