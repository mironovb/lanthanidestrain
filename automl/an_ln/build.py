#!/usr/bin/env python3
"""Build the lanthanide + americium/curium row table for the An/Ln transfer
experiments, reusing the lanthanide dataset pipeline function-for-function.

What changes relative to scripts/build_dataset_no3d.py:
  * the metal gate admits Am and Cm (the one-line filter in clean_and_dedup
    reads the module global LANTHANIDE_SET, which is patched here);
  * Am and Cm get metal descriptors: Z, Shannon CN-8 radius (Am 1.09 A;
    Cm 1.08 A, an estimate -- Shannon tabulates Cm only at CN 6), and a
    pseudo lanthanide_index obtained by placing the radius on the Ln radius
    scale (Am 5.27, Cm 5.93: the Pm gap), plus is_actinide = 1;
  * Am(VI) rows are dropped (the pipeline assumes M(III));
  * the geometry-spec stage is skipped (no Am structures), so the four plan
    columns are absent; the experiments drop them for lanthanides too;
  * output goes under automl/artifacts/an_ln/ (data/ is read-only).

Sanity: the lanthanide rows must come out identical in count (5,992) to the
shipped table; row identity is checked against final_ml_dataset_3d.parquet
by (metal, canonical_smiles, log_D) so the legacy ok_only mask and
composition_key can be joined for exact comparability of the Ln metric.

Usage:  PYTHONPATH=$PWD python3 -m automl.an_ln.build
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/artifacts/an_ln"
LN3D = REPO / "data/processed/final_ml_dataset_3d.parquet"

AN_DESCRIPTORS = {
    "Am": {"Atomic Number_metal": 95, "lanthanide_index": 5.27,
           "Ionic Radius_metal": 1.090},
    "Cm": {"Atomic Number_metal": 96, "lanthanide_index": 5.93,
           "Ionic Radius_metal": 1.080},
}


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "build_dataset_no3d", REPO / "scripts/build_dataset_no3d.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO))
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    B = load_pipeline()
    B.LANTHANIDE_SET = set(B.LANTHANIDE_SET) | set(AN_DESCRIPTORS)
    B.LANTHANIDE_DESCRIPTORS = {**B.LANTHANIDE_DESCRIPTORS, **AN_DESCRIPTORS}

    raw = B.read_raw(B.find_raw_dir(None))
    clean, mapping, counts = B.clean_and_dedup(raw)
    print(f"clean+dedup: {len(clean)} rows; per metal:")
    print(clean[B.METAL_COL].value_counts().to_string())

    # drop non-III actinides (Am(VI) etc.)
    ox_col = B.find_existing_column(clean, B.DEDUP_TEXT_ALIASES["metal_oxidation_state"])
    is_an = clean[B.METAL_COL].isin(AN_DESCRIPTORS)
    if ox_col is not None:
        ox = clean[ox_col].astype(str).str.strip().str.upper()
        bad = is_an & ~ox.isin({"III", "3", "NAN", "NONE", ""})
        print(f"dropping {int(bad.sum())} actinide rows with oxidation state "
              f"not III: {clean.loc[bad, ox_col].value_counts().to_dict()}")
        clean = clean[~bad].copy()

    clean, lig_rep = B.build_ligand_features(clean)
    clean, feat_rep = B.add_metal_and_condition_features(clean)
    ctx, _ = B.build_geometry_condition_context(clean)
    clean = pd.concat([clean.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)

    src = clean["source_file"].astype(str) if "source_file" in clean else "raw"
    exp = clean["exp_id"].astype(str) if "exp_id" in clean else clean.index.astype(str)
    clean["row_id"] = src.str.replace(".csv", "", regex=False) + ":" + exp
    clean["is_actinide"] = clean[B.METAL_COL].isin(AN_DESCRIPTORS).astype(int)

    keep_extra = ["row_id", "is_actinide", "geometry_environment_key",
                  "DOI", "Extractant_Name"] + list(ctx.columns)
    final, col_rep = B.select_final_columns(clean, extra_keep=keep_extra)
    for c in ("metal", "canonical_smiles", "log_D", "row_id", "is_actinide"):
        assert c in final.columns, f"missing {c}"

    # --- sanity vs the shipped lanthanide table --------------------------
    ln = final[final["is_actinide"] == 0]
    ref = pd.read_parquet(LN3D, columns=["safe_exp_id", "metal",
                                         "canonical_smiles", "log_D"])
    key = ["metal", "canonical_smiles", "log_D"]
    m = ln[key].round({"log_D": 6}).merge(
        ref.round({"log_D": 6}), on=key, how="left")
    print(f"\nlanthanide rows: {len(ln)} (shipped 5,992); matched to shipped "
          f"safe_exp_id by (metal, smiles, log_D): {m['safe_exp_id'].notna().sum()}")

    OUT.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT / "anln_rows.parquet", index=False)
    info = {"rows": int(len(final)),
            "per_metal": final["metal"].value_counts().to_dict(),
            "ligands": int(final["canonical_smiles"].nunique()),
            "an_descriptors": AN_DESCRIPTORS,
            "columns": int(final.shape[1]),
            "counts": {k: v for k, v in counts.items()
                       if isinstance(v, (int, float, str))}}
    (OUT / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "counts"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
