# Codex repository analysis and repair report

Date: 2026-07-16 (America/New_York)

## Scope and safety constraints

This pass inspected the current repository before changing code, limited fixes to confirmed and reproducible defects, and used only lightweight unit tests and syntax checks. No computational chemistry pipeline was run. No Slurm job was submitted. No data was deleted, and no generated XYZ file was opened for writing or overwritten.

The worktree was already dirty at the start of the pass. In particular, it contained modified scheduler/run reports and a tracked Python bytecode file, plus untracked regenerated geometry data, Slurm logs, and family-regeneration reports. Those items were treated as user-owned and were not intentionally changed by this work. The only intentional edits are:

- `scripts/build_unique_geometries.py`
- `scripts/build_dataset_no3d.py`
- `tests/test_missing_geometry_selection.py`
- `tests/test_stage_e.py`
- this report

## Architecture understood

The repository is a two-stage Python dataset pipeline with two small shared modules:

1. `scripts/build_dataset_no3d.py` is the dataset producer and optional 3D-feature consumer. It reads and cleans raw extraction data, canonicalizes ligands, builds ligand/condition/metal features, creates deterministic geometry specifications, and optionally joins clean geometry QC and external 3D feature blocks into an enriched dataset.
2. `src/chemistry/coordination.py` is the chemistry planning layer. It detects ligand donor atoms with RDKit, assigns metal-dependent coordination numbers, chooses ligand/fill stoichiometry, and creates deterministic complex specifications and build IDs.
3. `scripts/build_unique_geometries.py` is the geometry execution and recovery layer. It consumes frozen geometry specifications, isolates Architector builds in child processes, validates and QC-checks XYZ output, records crash-safe shard indexes, and supports recovery, merge, audit, and targeted regeneration workflows.
4. `src/geometry_schema.py` is the schema contract between geometry report producers and the Stage E consumer. It normalizes accepted-report path aliases and derives the `xyz_exists` gate.
5. `tests/` contains focused regression modules covering donor-family chemistry, regeneration planning/recovery, ligand-type overrides, missing-geometry selection, the geometry schema, and Stage E joins/gating.

Persistent data and reports live under `data/`, `reports/`, and `logs/`. The geometry builder is intentionally incremental and treats valid canonical XYZ files plus append-only index/report state as recovery inputs.

## Baseline verification

The default login-node Python (3.11.11) could not collect the suite because RDKit is not installed in that environment. This is an environment mismatch, not a confirmed repository defect. The existing repository-specific Conda environment at `/nfs/home/nshank3/.conda/envs/lanth/bin/python` contains RDKit 2026.03.3 but not pytest, so each test module was run directly using its lightweight standalone/unittest entry point.

Before changes, all existing standalone tests passed in that environment. This established a clean behavioral baseline but also showed that the defects below lacked regression coverage.

## Confirmed bug 1: malformed XYZ files were accepted as valid

### Evidence

`valid_xyz()` previously checked only that the first line was an integer, the declared atom count was at least two, and the file had enough lines. A file such as:

```text
2
comment
not coordinates
still not coordinates
```

therefore returned `True` even though neither atom row could be parsed. This affects correctness because `valid_xyz()` is used by missing-geometry selection and skip-existing/recovery paths. A corrupt file could be treated as completed geometry and omitted from rescue.

### Fix

`valid_xyz()` now checks every declared atom row for at least an atom label and three coordinates, converts the three coordinates to floats, and rejects NaN or infinite coordinates. It remains a deliberately cheap structural check and does not perform expensive chemistry or geometry analysis.

### Targeted verification

Added `test_valid_xyz_rejects_malformed_or_nonfinite_atom_rows`, covering both arbitrary non-coordinate rows and a NaN coordinate. Then ran `tests/test_missing_geometry_selection.py`: 11 tests passed in 0.129 seconds. Syntax compilation of the changed geometry script also passed.

## Confirmed bug 2: numeric-looking build IDs could disappear at the QC join

### Evidence

`load_geometry_qc_index()` used unconstrained CSV type inference. If a QC table contained only numeric-looking hash IDs, pandas inferred integers. Later, `attach_geometry_status()` converted dataset IDs to strings and compared them against the integer QC index, so a valid `OK` geometry silently failed to match. Casting after reading was insufficient for IDs with leading zeroes because inference had already discarded those zeroes.

### Fix

QC CSVs are now read with `build_id` explicitly typed as pandas `string`, preserving both identifier semantics and leading zeroes. The loader also canonicalizes the resulting column to the string extension dtype before concatenating tables.

### Targeted verification

Added `test_numeric_looking_build_id_survives_csv_type_inference` using the ID `001234567890`. The test verifies that the exact ID survives the CSV round trip and produces `geometry_ok=True`. Then ran `tests/test_stage_e.py`: 10 tests passed in 1.9 seconds.

## Final lightweight verification

All seven test modules were run directly with the repository's `lanth` interpreter after both fixes. Result: 48 tests passed, 0 failed.

- Coordination-family tests: 9 passed
- Family-regeneration plan tests: 1 passed
- Geometry-schema tests: 5 passed
- Ligand-type override tests: 8 passed
- Missing-geometry selection tests: 11 passed
- Regeneration-recovery tests: 4 passed
- Stage E tests: 10 passed

`python -m compileall -q` passed for each changed script during targeted verification, and `git diff --check` reported no whitespace errors.

## Outcome

Two reproducible correctness bugs were fixed one at a time with focused regression tests. No speculative chemistry behavior was changed. No heavy job, scheduler command, data deletion, or generated-XYZ write was performed.
