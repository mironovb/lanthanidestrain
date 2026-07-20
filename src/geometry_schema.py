"""Single source of truth for the geometry QC table schema.

``build_unique_geometries.py`` (the producer) writes several per-``build_id``
geometry tables; ``build_dataset_no3d.py`` (the consumer) reads them back to
decide which complexes are clean enough to carry 3D features. The two scripts
are coupled *only* by CSV column names -- a fragile contract that, left implicit,
produced a silent data-correctness bug: the stage-2 "accepted" table names its
paths ``accepted_*`` and carries no ``xyz_exists``, so once its rows were mixed
with the reports tables (which do have ``xyz_exists``) every rescued geometry was
quietly marked ``geometry_ok=False``.

Centralising the names here means a rename breaks loudly at import time on both
sides instead of silently mis-computing a correctness column downstream. The
normaliser reconciles the known producer aliases and *reports every fallback it
applies*, so the reconciliation is auditable in the run manifest rather than
invisible.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column names the consumer relies on.
# ---------------------------------------------------------------------------
BUILD_ID = "build_id"
QC_CLASS = "qc_class"
XYZ_EXISTS = "xyz_exists"
XYZ_PATH = "xyz_path"
MOL2_PATH = "mol2_path"
ENERGY_EV = "energy_eV"

# QC scalar columns as written into the geometry tables (first-shell donor
# distances + the gap to the next donor off the accepted geometry).
CORECN_MAX_DIST = "coreCN_max_dist"
NEXT_DONOR_DIST = "next_donor_dist"
GAP_AFTER_CORECN = "gap_after_coreCN"

# ---------------------------------------------------------------------------
# Producer aliases that must be normalised to the canonical names above. The
# stage-2 accepted table (REGEN_ACCEPTED_FIELDS in build_unique_geometries.py)
# names its geometry paths accepted_* and does not carry xyz_exists. The producer
# imports these same constants for its field list, so a rename here breaks both
# sides at once instead of only the reader.
# ---------------------------------------------------------------------------
ACCEPTED_XYZ_PATH = "accepted_xyz_path"
ACCEPTED_MOL2_PATH = "accepted_mol2_path"

PATH_ALIASES = {
    ACCEPTED_XYZ_PATH: XYZ_PATH,
    ACCEPTED_MOL2_PATH: MOL2_PATH,
}

# qc_class values that count as a usable clean 3D geometry.
CLEAN_QC_CLASSES = frozenset({"OK"})


def normalize_qc_table(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reconcile one geometry table to the canonical schema.

    Renames the known producer path aliases and derives ``xyz_exists`` when the
    table does not carry it (the accepted table does not). If no path column is
    available, the table is treated as not usable for clean geometry features.
    Returns a normalised
    *copy* plus a list of human-readable fallbacks applied, so the caller can
    record exactly how each source table was reconciled instead of guessing
    silently after a concat.
    """
    table = table.copy()
    applied: list[str] = []

    for src, dst in PATH_ALIASES.items():
        if dst not in table.columns and src in table.columns:
            table = table.rename(columns={src: dst})
            applied.append(f"renamed {src}->{dst}")

    if XYZ_EXISTS not in table.columns:
        if XYZ_PATH in table.columns:
            paths_txt = table[XYZ_PATH].astype("string").str.strip()
            table[XYZ_EXISTS] = (paths_txt.notna() & (paths_txt != "")).astype(bool)
            applied.append(f"derived {XYZ_EXISTS} from {XYZ_PATH}")
        else:
            table[XYZ_EXISTS] = False
            applied.append(f"defaulted {XYZ_EXISTS}=False (no path column)")

    return table, applied
