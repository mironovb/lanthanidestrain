#!/usr/bin/env python3
"""Assemble the AutoML feature matrix and define switchable feature blocks.

The baseline table ``data/processed/final_ml_dataset_3d.parquet`` is read-only.
This module joins it with the descriptors produced by ``automl.geom3d_features``
and exposes everything as *named blocks* so an ablation can attribute a change
in R^2 to a specific physical hypothesis rather than to "3D features" in bulk.

Blocks
------
2D baseline (already in the parquet):
    rdkit        RDKit ligand descriptors (MolWt, TPSA, logP, ...)
    ecfp         Morgan fingerprint bits
    metal        Z, lanthanide index, Shannon ionic radius
    cond         experimental conditions (acid, diluent, additive, T, conc.)

3D from the shipped feature blocks (already in the parquet):
    p3d_phys     feat3d__complex_physical__*   (xTB scalars, donor charges)
    p3d_poly     feat3d__polyhedron__*         (ordered donor distances/angles)

3D computed here (automl/artifacts/geom3d):
    g1 first_shell   g2 contraction   g3 polyhedron   g4 steric
    g5 electronic    g6 rdf           g7 global_shape g8 chelate  g9 topology

Derived here from the 3D blocks:
    g10 rel      within-ligand relative descriptors: every scalar 3D feature
                 re-expressed as (value - mean over the same ligand+anion
                 family) and as a rank across the lanthanide series.  These use
                 no target information, only geometry, so they are legal under
                 a leave-extractants-out split and they isolate exactly the
                 metal-selectivity signal that a per-ligand mean cannot carry.
    g11 pi       flattened GFN2-xTB persistence images (20x20) + PCA scores
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATASET_PATH = _REPO_ROOT / "data/processed/final_ml_dataset_3d.parquet"
GEOM3D_DIR = _REPO_ROOT / "automl/artifacts/geom3d"
PI_PATH = _REPO_ROOT / "data/processed/feature_blocks/complex_gfn2xtb_pi_images.npz"

TARGET = "log_D"
GROUP_COL = "extractant_group"

# Columns that must never become features (identifiers, provenance, target).
META_COLS = {
    "metal", "canonical_smiles", "LIGAND_SMILES", "extractant_name", "D", "log_D",
    "extractant_group", "split", "build_id", "geometry_key", "metal_symbol",
    "SMILES_FOR_ARCHITECTOR", "COORDLIST", "DONOR_TYPES", "inner_sphere_anion",
    "fill_ligand", "geometry_status", "geometry_note", "geometry_environment_key",
    "safe_exp_id", "geometry_qc_class", "geometry_source",
    "geometry_feature_build_id", "xyz_path", "mol2_path", "feature_block_manifest",
    "complex_pi_image_index", "vr_graph_index", "ligand_pi_control_image_index",
    "sample_weight_inv_metal_freq", "geometry_ok", "geometry_xtb_energy_eV",
}

# Condition columns kept as categorical context for the "composition" key.
GEOM_COND_COLS = [
    "geom_cond__acid_class", "geom_cond__acid_strength_bin",
    "geom_cond__nitrate_activity", "geom_cond__pH_bin",
    "geom_cond__diluent_family", "geom_cond__modifier_class",
    "geom_cond__temperature_bin", "geom_cond__contact_time_bin",
    "geom_cond__shaking_time_bin", "geom_cond__phase_ratio_bin",
]


# Ranked by grouped-CV permutation importance over the full 3D feature set.
# Everything below the cut had |importance| < 0.002 R^2 or a negative mean drop.
CORE_3D_FEATURES = [
    "g1__first_shell__donor_en_mean",      # realised donor electronegativity
    "g1__first_shell__donor_hard_frac",    # HSAB character of the bound shell
    "g1__first_shell__n_donor_O",
    "g1__first_shell__n_donor_N",
    "g1__first_shell__cn_observed",
    "g1__first_shell__d_mean",
    "g1__first_shell__shell_gap",
    "g5__electronic__q_metal",             # xTB charge left on the cation
    "g5__electronic__q_transfer",          # ligand -> metal charge transfer
    "g5__electronic__q_donor_mean",
    "g5__electronic__q_donor_std",
    "g5__electronic__coulomb_ML_sum",
    "g5__electronic__q_abs_sum",
    "g5__electronic__dipole_mag",
    "g5__electronic__force_rms",           # residual strain at the optimum
    "g2__contraction__d_over_r_mean",      # dimensionless cavity fit
    "g2__contraction__excess_mean",
    "g3__polyhedron__shell_anisotropy",    # open vs closed coordination
    "g3__polyhedron__cshm_best",
    "g4__steric__vbur_5p0",
    "g4__steric__cfrac_4p0_5p0",           # lipophilic second shell
    "g7__global_shape__sasa_apolar_frac",
]


@dataclass
class Blocks:
    """Named column groups, in the order they are offered to the AutoML."""
    mapping: dict[str, list[str]] = field(default_factory=dict)

    def add(self, name: str, cols: Iterable[str]) -> None:
        cols = [c for c in cols]
        if cols:
            self.mapping[name] = cols

    def select(self, names: Iterable[str]) -> list[str]:
        out: list[str] = []
        for n in names:
            out.extend(self.mapping.get(n, []))
        # de-duplicate, keep order
        seen, uniq = set(), []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def summary(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.mapping.items()}


# ---------------------------------------------------------------------------
# 3D descriptor loading
# ---------------------------------------------------------------------------
def load_geom3d(geom_dir: Path = GEOM3D_DIR) -> pd.DataFrame:
    """Concatenate every extraction shard into one geometry-level table."""
    shards = sorted(geom_dir.glob("geom3d_shard*.parquet"))
    if not shards:
        return pd.DataFrame(columns=["geometry_key"])
    frames = [pd.read_parquet(p) for p in shards]
    table = pd.concat(frames, ignore_index=True)
    return table.drop_duplicates("geometry_key").reset_index(drop=True)


def add_within_ligand_relative(geom: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Block g10: express every 3D scalar relative to its ligand+anion family.

    ``geometry_key`` is ``Z|ligand_smiles|anion``.  Grouping on
    ``ligand|anion`` collects the *same complex across the lanthanide series*.
    Subtracting the family mean removes everything that only says "which
    ligand is this" -- which the ECFP block already says perfectly -- and keeps
    only how this particular cation sits in that particular cavity.

    Legal under leave-extractants-out: the transform touches no target values,
    and for an unseen extractant the family is built from that extractant's own
    generated geometries.
    """
    if geom.empty:
        return geom
    merged = geom.merge(keys, on="geometry_key", how="left")
    fam = merged["ligand_anion_family"]
    num_cols = [c for c in geom.columns
                if c != "geometry_key" and pd.api.types.is_numeric_dtype(geom[c])]
    grouped = merged.groupby(fam)[num_cols]
    fam_mean = grouped.transform("mean")
    fam_std = grouped.transform("std")
    fam_size = merged.groupby(fam)[num_cols[0]].transform("size")

    out = pd.DataFrame({"geometry_key": merged["geometry_key"].values})
    delta = merged[num_cols].values - fam_mean.values
    zed = np.divide(delta, fam_std.values,
                    out=np.zeros_like(delta), where=(fam_std.values > 1e-12))
    # Rank of this metal within its family, normalised to [0, 1].
    rank = grouped.rank(pct=True).values

    single = (fam_size.values <= 1)
    for arr, tag in ((delta, "d"), (zed, "z"), (rank, "r")):
        block = arr.copy()
        block[single, :] = np.nan  # a one-member family carries no relative info
        for j, c in enumerate(num_cols):
            out[f"g10__rel_{tag}__{c}"] = block[:, j]
    out["g10__rel__family_size"] = fam_size.values.astype(float)
    return out


def add_series_smoothed(geom: pd.DataFrame, keys: pd.DataFrame,
                        min_members: int = 4) -> pd.DataFrame:
    """Blocks g12/g13/g14: denoise each descriptor along the lanthanide series.

    Motivation (measured, not assumed).  In the first ablation every raw 3D
    block raised overall and within-extractant R^2 but *lowered* the pure
    metal-selectivity metrics -- Spearman of the predicted La..Lu order fell
    from 0.63 to ~0.49 and the pairwise log-SF R^2 from 0.41 to ~0.28.  The
    baseline's clean, strictly monotone lanthanide descriptors (Z, index,
    Shannon radius) were being replaced by geometry proxies that carry the same
    trend *plus* conformer noise.  Each geometry is one stochastic Architector
    conformer optimised with GFN2-xTB; the conformational scatter in a M-O
    distance is easily 0.05 A, while the radius step between adjacent
    lanthanides is only ~0.013 A.  So the raw descriptor is mostly noise at the
    scale that matters for separation.

    The fix is to fit, inside each ligand+anion family, a straight line of every
    descriptor against the Shannon ionic radius, and hand the model the fit
    rather than the raw value:

      g12 ``smooth``  the fitted value  -- the systematic size response, denoised
      g13 ``slope``   the family slope  -- "how strongly does this cavity react
                      to cation size", a ligand-level selectivity descriptor
                      that is identical for every metal in the family
      g14 ``fmean``   the family mean   -- a purely ligand-shaped 3D descriptor
                      with the metal dependence integrated out

    Only geometry enters the fit; no target value is used, and a family is built
    entirely from its own extractant's structures, so this stays legal under a
    leave-extractants-out split.
    """
    if geom.empty:
        return pd.DataFrame(columns=["geometry_key"])
    merged = geom.merge(keys, on="geometry_key", how="left")
    fam = merged["ligand_anion_family"].astype(str)
    radius_col = "g2__contraction__ionic_radius"
    if radius_col not in merged.columns:
        return pd.DataFrame({"geometry_key": merged["geometry_key"]})
    r = merged[radius_col].to_numpy(dtype=float)

    num_cols = [c for c in geom.columns
                if c != "geometry_key" and pd.api.types.is_numeric_dtype(geom[c])]
    values = merged[num_cols].to_numpy(dtype=float)

    smooth = np.full_like(values, np.nan)
    slope = np.full_like(values, np.nan)
    fmean = np.full_like(values, np.nan)
    fitr2 = np.full_like(values, np.nan)

    for _, idx in pd.Series(np.arange(len(merged))).groupby(fam.to_numpy()).groups.items():
        idx = np.asarray(list(idx))
        v = values[idx]
        rr = r[idx]
        col_mean = np.nanmean(v, axis=0)
        fmean[idx] = col_mean
        ok_r = np.isfinite(rr)
        if idx.size < min_members or ok_r.sum() < min_members:
            smooth[idx] = col_mean
            continue
        # Per column least squares against the ionic radius, skipping NaNs.
        for j in range(v.shape[1]):
            m = ok_r & np.isfinite(v[:, j])
            if m.sum() < min_members:
                smooth[idx, j] = col_mean[j]
                continue
            x, yv = rr[m], v[m, j]
            if np.ptp(x) < 1e-9 or np.ptp(yv) < 1e-12:
                smooth[idx, j] = yv.mean()
                slope[idx, j] = 0.0
                fitr2[idx, j] = 0.0
                continue
            b, a = np.polyfit(x, yv, 1)
            pred = a + b * rr
            smooth[idx, j] = pred
            slope[idx, j] = b
            ss_tot = float(np.sum((yv - yv.mean()) ** 2))
            ss_res = float(np.sum((yv - (a + b * x)) ** 2))
            fitr2[idx, j] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    frames = {"geometry_key": merged["geometry_key"].to_numpy()}
    out = pd.DataFrame(frames)
    parts = [out]
    for arr, tag in ((smooth, "g12__smooth"), (slope, "g13__slope"),
                     (fmean, "g14__fmean"), (fitr2, "g13__fitr2")):
        parts.append(pd.DataFrame(
            arr, columns=[f"{tag}__{c}" for c in num_cols], index=out.index))
    return pd.concat(parts, axis=1)


def add_cn_free(geom: pd.DataFrame) -> pd.DataFrame:
    """Block g15: remove the artificial coordination-number step.

    Measured cause, not a guess.  ``src/chemistry/coordination.cn_for_Z`` assigns
    CN 9 to La-Gd and CN 8 to Tb-Lu, so every generated geometry inherits a hard
    discontinuity in the middle of the series.  Averaged over families the jump
    at the Gd -> Tb boundary is 10x the typical adjacent-metal step for
    ``cn_observed``, 17x for the donor-hull volume, 7x for the mean M-L distance
    and 5x for %V_bur.  The *measured* log D has no such step: its Gd -> Tb
    change (-0.064) is smaller than the median adjacent step (0.140).

    So a model given raw 3D descriptors is handed a staircase where the
    experiment shows a ramp, and it duly predicts a staircase -- which is
    precisely how the pairwise separation-factor R^2 and the series Spearman
    fall when 3D blocks are switched on.

    The one block that does *not* carry the artefact is the xTB electronic
    block: ``q_metal`` has a Gd -> Tb jump ratio of 0.84, i.e. no step at all.
    That explains why G5 is the only raw block that improves the model.

    This function regresses every descriptor on the observed coordination number
    (globally, across all geometries) and keeps the residual, so the CN main
    effect -- and with it the imposed staircase -- is removed while the
    ligand- and metal-specific variation survives.  A bond-valence sum is added
    as an explicitly CN-invariant measure of bonding strength.
    """
    if geom.empty or "g1__first_shell__cn_observed" not in geom.columns:
        return pd.DataFrame(columns=["geometry_key"])
    cn = geom["g1__first_shell__cn_observed"].to_numpy(dtype=float)
    num_cols = [c for c in geom.columns
                if c != "geometry_key" and pd.api.types.is_numeric_dtype(geom[c])
                and not c.startswith(("g10__", "g11__", "g12__", "g13__", "g14__"))]
    out = pd.DataFrame({"geometry_key": geom["geometry_key"].to_numpy()})
    cols: dict[str, np.ndarray] = {}
    for c in num_cols:
        v = geom[c].to_numpy(dtype=float)
        m = np.isfinite(v) & np.isfinite(cn)
        resid = np.full(len(v), np.nan)
        if m.sum() >= 8 and np.ptp(cn[m]) > 0 and np.ptp(v[m]) > 1e-12:
            b, a = np.polyfit(cn[m], v[m], 1)
            resid[m] = v[m] - (a + b * cn[m])
        else:
            resid[m] = v[m] - np.mean(v[m])
        cols[f"g15__cnfree__{c}"] = resid
    # Intensive (per-donor) forms of the extensive descriptors.
    for c, name in (("g3__polyhedron__hull_volume", "hull_volume_per_donor"),
                    ("g3__polyhedron__hull_area", "hull_area_per_donor"),
                    ("g5__electronic__q_donor_sum", "q_donor_per_donor"),
                    ("g4__steric__vbur_5p0", "vbur_per_donor")):
        if c in geom.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"g15__intensive__{name}"] = (
                    geom[c].to_numpy(dtype=float) / np.where(cn > 0, cn, np.nan))
    # Bond-valence sum: sum(exp((R0 - d)/0.37)).  Classic CN-invariant measure of
    # how much bonding the cation actually receives.  Reconstructed from the mean
    # distance and the observed CN, using an O-donor R0 for Ln(III).
    if "g1__first_shell__d_mean" in geom.columns:
        d = geom["g1__first_shell__d_mean"].to_numpy(dtype=float)
        cols["g15__intensive__bond_valence_sum"] = cn * np.exp((2.10 - d) / 0.37)
    return pd.concat([out, pd.DataFrame(cols, index=out.index)], axis=1)


def load_pi_images(df: pd.DataFrame) -> pd.DataFrame:
    """Block g11: the shipped 20x20 GFN2-xTB persistence images, flattened.

    The manifest marks these as image-native (CNN/ViT readout).  For a tabular
    AutoML we additionally offer them flattened plus a compact PCA projection,
    and record which representation wins.
    """
    if not PI_PATH.exists() or "complex_pi_image_index" not in df.columns:
        return pd.DataFrame(index=df.index)
    with np.load(PI_PATH) as npz:
        key = "images" if "images" in npz.files else npz.files[0]
        images = npz[key]
    flat = images.reshape(len(images), -1)
    idx = df["complex_pi_image_index"].to_numpy()
    out = np.full((len(df), flat.shape[1]), np.nan)
    valid = pd.notna(idx)
    ii = idx[valid].astype(int)
    inside = ii < len(flat)
    rows = np.flatnonzero(valid)[inside]
    out[rows] = flat[ii[inside]]
    cols = [f"g11__pi__px_{k:03d}" for k in range(flat.shape[1])]
    frame = pd.DataFrame(out, columns=cols, index=df.index)
    # Drop all-zero pixels (persistence images are sparse).
    keep = frame.columns[np.nanstd(frame.values, axis=0) > 1e-9]
    return frame[keep]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_matrix(require_3d: bool = True,
                 geom_dir: Path = GEOM3D_DIR,
                 include_pi: bool = True) -> tuple[pd.DataFrame, Blocks, dict[str, Any]]:
    """Return (dataframe, block map, provenance info)."""
    df = pd.read_parquet(DATASET_PATH)
    info: dict[str, Any] = {"rows_total": int(len(df))}

    # ligand|anion family key for the relative block, and a metal-free
    # composition key used by the selectivity metrics.
    parts = df["geometry_key"].str.split("|", n=2, expand=True)
    df["_geom_Z"] = pd.to_numeric(parts[0], errors="coerce")
    df["ligand_anion_family"] = parts[1].fillna("") + "|" + parts[2].fillna("")
    cond_key = df[GEOM_COND_COLS].astype(str).agg("|".join, axis=1)
    df["composition_key"] = df[GROUP_COL].astype(str) + "||" + cond_key
    # Strict variant: every numeric condition included, so the *only* thing that
    # varies inside a block is the lanthanide.  Delta learning needs this --
    # with the binned key two rows can share a block while differing in
    # extractant concentration, which turns a real log D difference into label
    # noise on a zero-difference feature vector.
    cond_num = [c for c in df.columns if c.startswith("cond__")]
    strict = (df[cond_num].round(6).astype(str).agg("|".join, axis=1)
              if cond_num else pd.Series("", index=df.index))
    df["strict_composition_key"] = (df[GROUP_COL].astype(str) + "||"
                                    + cond_key + "||" + strict)

    geom = load_geom3d(geom_dir)
    info["geom3d_geometries"] = int(len(geom))
    if not geom.empty:
        fam_keys = df[["geometry_key", "ligand_anion_family"]].drop_duplicates("geometry_key")
        rel = add_within_ligand_relative(geom, fam_keys)
        smoothed = add_series_smoothed(geom, fam_keys)
        cnfree = add_cn_free(geom)
        geom = geom.merge(rel, on="geometry_key", how="left")
        geom = geom.merge(smoothed, on="geometry_key", how="left")
        geom = geom.merge(cnfree, on="geometry_key", how="left")
        df = df.merge(geom, on="geometry_key", how="left")

    if include_pi:
        pi = load_pi_images(df)
        if len(pi.columns):
            df = pd.concat([df, pi], axis=1)

    # Geometry-quality context.  Descriptors are now computed for BORDERLINE and
    # FAIL_LONG_BOND geometries too (99% row coverage instead of 79%), so the
    # model must be told how trustworthy each geometry is rather than silently
    # treating a long-bond artefact as a real bond.
    qc = pd.get_dummies(df["geometry_qc_class"].astype("string").fillna("MISSING"),
                        prefix="qc__class").astype(np.int8)
    df = pd.concat([df, qc], axis=1)
    df["qc__geometry_ok"] = df["geometry_ok"].astype(np.int8)

    # ---- block definitions -------------------------------------------------
    blocks = Blocks()
    cols = list(df.columns)
    blocks.add("rdkit", ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors",
                         "NumRotatableBonds", "NumAromaticRings",
                         "NumAliphaticRings", "RingCount", "FractionCSP3", "MolLogP"])
    blocks.add("ecfp", [c for c in cols if c.startswith("ecfp_")])
    blocks.add("metal", ["Atomic Number_metal", "lanthanide_index",
                         "Ionic Radius_metal", "metal_ox"])
    blocks.add("cond", [c for c in cols if c.startswith("cond__")])
    blocks.add("plan", ["DENTATE", "coreCN", "n_ligs", "n_fill"])
    blocks.add("p3d_phys", [c for c in cols if c.startswith("feat3d__complex_physical__")])
    blocks.add("p3d_poly", [c for c in cols if c.startswith("feat3d__polyhedron")])
    for tag, name in (("g1", "first_shell"), ("g2", "contraction"), ("g3", "polyhedron"),
                      ("g4", "steric"), ("g5", "electronic"), ("g6", "rdf"),
                      ("g7", "global_shape"), ("g8", "chelate"), ("g9", "topology")):
        blocks.add(tag, [c for c in cols if c.startswith(f"{tag}__{name}__")])
    blocks.add("g10", [c for c in cols if c.startswith("g10__")])
    blocks.add("g11", [c for c in cols if c.startswith("g11__")])
    blocks.add("g12", [c for c in cols if c.startswith("g12__")])
    blocks.add("g13", [c for c in cols if c.startswith("g13__")])
    blocks.add("g14", [c for c in cols if c.startswith("g14__")])
    blocks.add("g15", [c for c in cols if c.startswith("g15__")])
    core = ("first_shell", "contraction", "polyhedron", "steric", "electronic",
            "global_shape", "chelate")
    blocks.add("g15c", [c for c in cols if c.startswith("g15__")
                        and (c.startswith("g15__intensive__")
                             or any(f"__{k}__" in c for k in core))])
    blocks.add("qc", [c for c in cols if c.startswith("qc__")])
    # Curated micro-block.  Chosen by grouped-CV permutation importance on the
    # full 3D set (automl/artifacts/selection/importance_lgbm_has3d.csv), not by
    # inspection: these are the only 3D columns whose permutation drop exceeded
    # 0.002 R^2.  Chemically they are one coherent story -- *which* donors the
    # optimised structure actually places in the first shell, how hard/
    # electronegative that realised donor set is, and how much charge the ligand
    # transfers to the cation.  None of it is readable from the 2D graph, which
    # only lists donors that *could* bind.
    blocks.add("g_core", [c for c in CORE_3D_FEATURES if c in cols])
    # Compact, curated variants: the same denoising restricted to the physically
    # interpretable scalars, leaving out the 128 RDF bins and the persistence
    # summaries so a small model can use them without dilution.
    core = ("first_shell", "contraction", "polyhedron", "steric", "electronic",
            "global_shape", "chelate")
    blocks.add("g12c", [c for c in cols if c.startswith("g12__")
                        and any(f"__{k}__" in c for k in core)])
    blocks.add("g13c", [c for c in cols if c.startswith("g13__slope__")
                        and any(f"__{k}__" in c for k in core)])
    blocks.add("g14c", [c for c in cols if c.startswith("g14__")
                        and any(f"__{k}__" in c for k in core)])

    # Sanitise infinities.  Two *shipped* columns carry +inf --
    # `feat3d__polyhedron_scalars__coreCN_donor_gap` (7 rows) and
    # `next_donor_dist` (6 rows) -- which is the encoding for "there is no next
    # donor beyond the coordination shell".  NaN says that honestly; +inf makes
    # XGBoost refuse the matrix outright and gives every other tree learner a
    # meaningless split point.  The source parquet is left untouched.
    all_feat = blocks.select(blocks.mapping.keys())
    numeric_feat = [c for c in all_feat if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_feat:
        block = df[numeric_feat].to_numpy(dtype=float, copy=True)
        n_inf = int(np.isinf(block).sum())
        if n_inf:
            block[np.isinf(block)] = np.nan
            df[numeric_feat] = block
        info["infinities_replaced_with_nan"] = n_inf

    # Drop columns that are entirely null or constant -- they cost time and can
    # break scalers, and their absence never changes a tree model.
    dead = [c for c in all_feat
            if df[c].isna().all() or df[c].nunique(dropna=True) <= 1]
    if dead:
        dead_set = set(dead)
        blocks.mapping = {k: [c for c in v if c not in dead_set]
                          for k, v in blocks.mapping.items()}
        blocks.mapping = {k: v for k, v in blocks.mapping.items() if v}
    info["dropped_constant_or_empty"] = len(dead)

    # 3D availability flag + optional row restriction.
    has3d_probe = "g1__first_shell__cn_observed"
    df["has_3d"] = df[has3d_probe].notna() if has3d_probe in df.columns else False
    info["rows_with_3d"] = int(df["has_3d"].sum())
    if require_3d:
        df = df[df["has_3d"]].reset_index(drop=True)
    info["rows_used"] = int(len(df))
    info["n_groups"] = int(df[GROUP_COL].nunique())
    info["blocks"] = blocks.summary()
    return df, blocks, info


BASE_2D = ("rdkit", "ecfp", "metal", "cond", "plan")
NEW_3D_BLOCKS = ("g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10")
ALL_3D_BLOCKS = ("p3d_phys", "p3d_poly") + NEW_3D_BLOCKS + ("g11",)

BLOCK_PRESETS: dict[str, tuple[str, ...]] = {
    # --- reference points ---------------------------------------------------
    "baseline_2d":      BASE_2D,
    "baseline_no_ecfp": ("rdkit", "metal", "cond", "plan"),
    "baseline_2d_qc":   BASE_2D + ("qc",),
    # --- shipped 3D blocks --------------------------------------------------
    "plus_p3d_phys":    BASE_2D + ("qc", "p3d_phys"),
    "plus_p3d_poly":    BASE_2D + ("qc", "p3d_poly"),
    "plus_p3d_all":     BASE_2D + ("qc", "p3d_phys", "p3d_poly"),
    # --- one new block at a time (attribution) ------------------------------
    **{f"plus_{b}": BASE_2D + ("qc", b)
       for b in NEW_3D_BLOCKS + ("g11", "g12", "g13", "g14",
                                 "g12c", "g13c", "g14c")},
    # --- denoised-series hypothesis ----------------------------------------
    # g5 was the strongest raw block; g13/g14 are metal-free, so they should
    # raise R2_between without touching the series ordering; g12 replaces the
    # noisy raw values with the fitted size response.
    "denoised":        BASE_2D + ("qc", "g12c", "g13c", "g14c"),
    "denoised_g5":     BASE_2D + ("qc", "g5", "g12c", "g13c", "g14c"),
    "ligand3d_only":   BASE_2D + ("qc", "g13c", "g14c"),
    "electronic_plus": BASE_2D + ("qc", "g5", "g14c"),
    "best_guess":      BASE_2D + ("qc", "g5", "g13c", "g14c"),
    # --- CN-artefact correction --------------------------------------------
    "plus_g15":        BASE_2D + ("qc", "g15"),
    "plus_g15c":       BASE_2D + ("qc", "g15c"),
    "cnfree":          BASE_2D + ("qc", "g5", "g15c"),
    "cnfree_full":     BASE_2D + ("qc", "g5", "g15c", "g13c", "g14c"),
    # --- complementary combination -----------------------------------------
    # Measured: g15c gives the best between-extractant R2 (0.716) because the
    # CN staircase is gone; g14c/g13c give the best within-extractant R2 (0.298)
    # and are the only 3D blocks that leave the La->Lu ordering intact, because
    # they carry no per-metal conformer noise at all.  Combine the two.
    "cnfree_ligand":     BASE_2D + ("qc", "g15c", "g14c"),
    "cnfree_ligand_g13": BASE_2D + ("qc", "g15c", "g13c", "g14c"),
    "g5_ligand":         BASE_2D + ("qc", "g5", "g14c"),
    "g5_ligand_g13":     BASE_2D + ("qc", "g5", "g13c", "g14c"),
    "core_ligand_cnfree": BASE_2D + ("qc", "g_core", "g14c", "g15c"),
    # --- curated micro-block (importance-selected) --------------------------
    "core3d":          BASE_2D + ("g_core",),
    "core3d_qc":       BASE_2D + ("qc", "g_core"),
    "core3d_g5":       BASE_2D + ("qc", "g_core", "g5"),
    "core3d_ligand":   BASE_2D + ("qc", "g_core", "g13c", "g14c"),
    "core3d_smooth":   BASE_2D + ("qc", "g_core", "g12c"),
    "core3d_all":      BASE_2D + ("qc", "g_core", "g12c", "g13c", "g14c"),
    "core3d_cnfree":   BASE_2D + ("qc", "g_core", "g15c"),
    "core3d_cnfree_lig": BASE_2D + ("qc", "g_core", "g15c", "g14c"),
    # --- physically motivated combinations ----------------------------------
    "inner_sphere":  BASE_2D + ("qc", "g1", "g2", "g3", "g8"),
    "outer_sphere":  BASE_2D + ("qc", "g4", "g7"),
    "electronic":    BASE_2D + ("qc", "g5"),
    "fingerprint3d": BASE_2D + ("qc", "g6", "g9", "g11"),
    "selectivity":   BASE_2D + ("qc", "g2", "g10"),
    # --- leave-one-block-out over the new 3D blocks -------------------------
    **{f"drop_{b}": BASE_2D + ("qc",) + tuple(x for x in NEW_3D_BLOCKS if x != b)
       for b in NEW_3D_BLOCKS},
    # --- everything ---------------------------------------------------------
    "all_new_3d":     BASE_2D + ("qc",) + NEW_3D_BLOCKS,
    "all_3d":         BASE_2D + ("qc",) + ALL_3D_BLOCKS,
    "all_3d_no_ecfp": ("rdkit", "metal", "cond", "plan", "qc") + ALL_3D_BLOCKS,
    "only_3d":        ("metal", "cond", "qc") + ALL_3D_BLOCKS,
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-3d", action="store_true", default=True)
    ap.add_argument("--all-rows", dest="require_3d", action="store_false")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    df, blocks, info = build_matrix(require_3d=args.require_3d)
    print(json.dumps(info, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        with open(Path(args.out).with_suffix(".blocks.json"), "w") as fh:
            json.dump({"blocks": blocks.mapping, "info": info}, fh, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
