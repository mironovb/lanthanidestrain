#!/bin/bash
#SBATCH --job-name=ln_geom_pipe
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --chdir=/nfs/home/nshank3/lanthanide_dataset_builder
#SBATCH --output=logs/ln_geom_pipe_%A_%a.out
#SBATCH --error=logs/ln_geom_pipe_%A_%a.err

set -euo pipefail

# One entrypoint for the cluster geometry chain.
#
# Submit from the cluster login node:
#   bash slurm/geometry_pipeline.slurm submit
#
# Useful dry run:
#   DRY_RUN=1 bash slurm/geometry_pipeline.slurm submit
#
# The same file is submitted back to SLURM as a worker with PIPELINE_STEP set to:
#   build       - full geometry shard
#   post_qc     - recover/merge/audit/rebuild nearest-coreCN queues
#   regen       - focused FAIL_LONG_BOND regeneration shard
#   merge_regen - merge regeneration reports
#   all_remaining_status - reconcile every source row after all route merges
#   gradual_followup - validate the adaptive merge and submit the bounded follow-up

PROJECT_DIR="${PROJECT_DIR:-/nfs/home/nshank3/lanthanide_dataset_builder}"
ACCOUNT="${ACCOUNT:-acf-utk0011}"
PARTITION="${PARTITION:-campus}"
ARCH_ENV="${ARCH_ENV:-$HOME/.conda/envs/arch}"
PY="${PY:-$ARCH_ENV/bin/python}"

NUM_SHARDS="${NUM_SHARDS:-16}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-8}"
REGEN_NUM_SHARDS="${REGEN_NUM_SHARDS:-36}"
REGEN_MAX_CONCURRENT_TASKS="${REGEN_MAX_CONCURRENT_TASKS:-6}"
BUILD_WALLTIME="${BUILD_WALLTIME:-09:00:00}"
POST_QC_WALLTIME="${POST_QC_WALLTIME:-01:00:00}"
REGEN_WALLTIME="${REGEN_WALLTIME:-12:00:00}"
MERGE_REGEN_WALLTIME="${MERGE_REGEN_WALLTIME:-00:30:00}"

BUILD_CPUS="${BUILD_CPUS:-4}"
BUILD_MEM="${BUILD_MEM:-2G}"
POST_QC_CPUS="${POST_QC_CPUS:-1}"
POST_QC_MEM="${POST_QC_MEM:-2G}"
REGEN_CPUS="${REGEN_CPUS:-4}"
REGEN_MEM="${REGEN_MEM:-2G}"
MERGE_REGEN_CPUS="${MERGE_REGEN_CPUS:-1}"
MERGE_REGEN_MEM="${MERGE_REGEN_MEM:-1G}"

BUILD_MAX_ATTEMPTS="${BUILD_MAX_ATTEMPTS:-3}"
BUILD_TIMEOUT_PER_COMPLEX="${BUILD_TIMEOUT_PER_COMPLEX:-7200}"
REGEN_MAX_ATTEMPTS="${REGEN_MAX_ATTEMPTS:-2}"
REGEN_TIMEOUT_PER_COMPLEX="${REGEN_TIMEOUT_PER_COMPLEX:-2400}"
REGEN_WALLTIME_MARGIN_SECONDS="${REGEN_WALLTIME_MARGIN_SECONDS:-1800}"
REGEN_SEED="${REGEN_SEED:-61453}"
REGEN_SEED_STEP="${REGEN_SEED_STEP:-7919}"
REGEN_PROFILE_SEQUENCE="${REGEN_PROFILE_SEQUENCE:-standard}"
REGEN_REPORTS_DIR="${REGEN_REPORTS_DIR:-reports}"
REGEN_OUT="${REGEN_OUT:-data/processed/geometries_regenerated_fail_long_bond}"
REGEN_RUN_ID="${REGEN_RUN_ID:-}"
REGEN_QUEUE_SHA256="${REGEN_QUEUE_SHA256:-}"
REGEN_N_SYMMETRIES="${REGEN_N_SYMMETRIES:-40}"
REGEN_N_SYMMETRIES_STEP="${REGEN_N_SYMMETRIES_STEP:-10}"
REGEN_N_CONFORMERS="${REGEN_N_CONFORMERS:-5}"
REGEN_N_CONFORMERS_STEP="${REGEN_N_CONFORMERS_STEP:-1}"
REGEN_XTB_MAX_ITERATIONS="${REGEN_XTB_MAX_ITERATIONS:-250}"
REGEN_LONG_BOND_THRESHOLD="${REGEN_LONG_BOND_THRESHOLD:-3.10}"
REGEN_BORDERLINE_LONGISH_THRESHOLD="${REGEN_BORDERLINE_LONGISH_THRESHOLD:-2.95}"
REGEN_AMBIGUOUS_GAP_THRESHOLD="${REGEN_AMBIGUOUS_GAP_THRESHOLD:-0.10}"
SEED="${SEED:-$REGEN_SEED}"
SEED_STEP="${SEED_STEP:-$REGEN_SEED_STEP}"
PROFILE_SEQUENCE="${PROFILE_SEQUENCE:-$REGEN_PROFILE_SEQUENCE}"
MAX_TOTAL_SECONDS="${MAX_TOTAL_SECONDS:-0}"
SPECS_CSV="${SPECS_CSV:-data/processed/geometry_specs.csv}"
FIXED_MISSING_QUEUE="${FIXED_MISSING_QUEUE:-0}"
INDEX_TAG="${INDEX_TAG:-}"

RETRY_FAILED="${RETRY_FAILED:-1}"
RETRY_STATUS_LIST="${RETRY_STATUS_LIST:-}"
RETRY_ONLY="${RETRY_ONLY:-0}"
MISSING_ONLY="${MISSING_ONLY:-0}"
HARD_LIGAND_MODE="${HARD_LIGAND_MODE:-0}"
SUBMIT_REGEN_AFTER_QC="${SUBMIT_REGEN_AFTER_QC:-1}"
POST_BUILD_DEPENDENCY="${POST_BUILD_DEPENDENCY:-afterok}"
VERIFY_MISSING_COMPLETE="${VERIFY_MISSING_COMPLETE:-0}"
INPUT_CSV="${INPUT_CSV:-reports/geometry_regenerate_fail_long_bond.csv}"
EXPECTED_INPUT_ROWS="${EXPECTED_INPUT_ROWS:-}"
LIGTYPE_OVERRIDES="${LIGTYPE_OVERRIDES:-data/processed/ligtype_overrides.csv}"
LIGTYPE_RESCUE_OVERRIDES="${LIGTYPE_RESCUE_OVERRIDES:-reports/geometry_ligtype_rescue_queue.csv}"

MISSING_REGEN_SOURCE_INPUT="${MISSING_REGEN_SOURCE_INPUT:-reports/geometry_regenerate_fail_long_bond.csv}"
MISSING_REGEN_STILL_FAILED="${MISSING_REGEN_STILL_FAILED:-reports/regenerated_fail_long_bond_still_failed.csv}"
MISSING_REGEN_QUEUE="${MISSING_REGEN_QUEUE:-reports/geometry_regenerate_missing_geometry_rescue.csv}"
MISSING_REGEN_LIGTYPE_OVERRIDES="${MISSING_REGEN_LIGTYPE_OVERRIDES:-reports/geometry_regenerate_missing_geometry_ligtype_overrides.csv}"
MISSING_REGEN_REPORTS_DIR="${MISSING_REGEN_REPORTS_DIR:-reports/missing_geometry_rescue}"
MISSING_REGEN_EXPECTED_ROWS="${MISSING_REGEN_EXPECTED_ROWS:-}"
MISSING_REGEN_NUM_SHARDS="${MISSING_REGEN_NUM_SHARDS:-12}"
MISSING_REGEN_MAX_CONCURRENT_TASKS="${MISSING_REGEN_MAX_CONCURRENT_TASKS:-4}"
MISSING_REGEN_MAX_ATTEMPTS="${MISSING_REGEN_MAX_ATTEMPTS:-3}"
MISSING_REGEN_TIMEOUT_PER_COMPLEX="${MISSING_REGEN_TIMEOUT_PER_COMPLEX:-5400}"
MISSING_REGEN_SEED="${MISSING_REGEN_SEED:-104729}"
MISSING_REGEN_SEED_STEP="${MISSING_REGEN_SEED_STEP:-15485863}"
MISSING_REGEN_PROFILE_SEQUENCE="${MISSING_REGEN_PROFILE_SEQUENCE:-large_ligand_fast:standard:large_ligand_deep}"
MISSING_REGEN_MEM="${MISSING_REGEN_MEM:-4G}"

# Chemistry-family regeneration is planned into bounded, non-overlapping
# queues. Submit one group at a time so the rescue stays observable and within
# the normal QOS envelope.
FAMILY_PLAN_DIR="${FAMILY_PLAN_DIR:-reports/family_regeneration}"
FAMILY_GROUP="${FAMILY_GROUP:-template_pincer}"
FAMILY_REPORTS_ROOT="${FAMILY_REPORTS_ROOT:-reports/family_regeneration_runs}"
FAMILY_OUT_ROOT="${FAMILY_OUT_ROOT:-data/processed/geometries_family_regenerated}"
FAMILY_LIGTYPE_OVERRIDES="${FAMILY_LIGTYPE_OVERRIDES:-$FAMILY_PLAN_DIR/template_ligtype_overrides.csv}"
FAMILY_MAX_ROWS="${FAMILY_MAX_ROWS:-}"
FAMILY_PLAN_PREPARED="${FAMILY_PLAN_PREPARED:-0}"
FAMILY_DEPENDENCY_CHAIN="${FAMILY_DEPENDENCY_CHAIN:-0}"

# One immutable, fully preflighted submission for the sources already marked
# unfinished by current reports/index state. Their candidate geometries are
# rechecked with the composition + nearest-coreCN gate. A fresh plan ID
# isolates queues, reports, and generated geometries. Set
# ALL_REMAINING_PLAN_PREPARED=1 with explicit directories only to resume an
# already-audited immutable plan.
ALL_REMAINING_PLAN_ROOT="${ALL_REMAINING_PLAN_ROOT:-reports/all_remaining_regeneration_plans}"
ALL_REMAINING_RUNS_ROOT="${ALL_REMAINING_RUNS_ROOT:-reports/all_remaining_regeneration_runs}"
ALL_REMAINING_OUT_ROOT="${ALL_REMAINING_OUT_ROOT:-data/processed/geometries_all_remaining}"
ALL_REMAINING_PLAN_ID="${ALL_REMAINING_PLAN_ID:-}"
ALL_REMAINING_PLAN_DIR="${ALL_REMAINING_PLAN_DIR:-}"
ALL_REMAINING_RUNS_DIR="${ALL_REMAINING_RUNS_DIR:-}"
ALL_REMAINING_OUT_DIR="${ALL_REMAINING_OUT_DIR:-}"
ALL_REMAINING_HYPOTHESIS_VERSION="${ALL_REMAINING_HYPOTHESIS_VERSION:-all_remaining_v1}"
ALL_REMAINING_SCOPE="${ALL_REMAINING_SCOPE:-known-unfinished}"
ALL_REMAINING_PLAN_PREPARED="${ALL_REMAINING_PLAN_PREPARED:-0}"
ALL_REMAINING_STATUS_WALLTIME="${ALL_REMAINING_STATUS_WALLTIME:-00:30:00}"
ALL_REMAINING_STATUS_MEM="${ALL_REMAINING_STATUS_MEM:-2G}"

# Bounded adjacent-CN/fill hypotheses for repeated compact ambiguous shells.
# This is intentionally isolated from the legacy FAIL_LONG_BOND reports and
# output tree so the source hypothesis and geometry remain reviewable.
ADAPTIVE_REGEN_INPUT="${ADAPTIVE_REGEN_INPUT:-$FAMILY_REPORTS_ROOT/template_pincer/regenerated_fail_long_bond_still_failed.csv}"
ADAPTIVE_REGEN_QUEUE="${ADAPTIVE_REGEN_QUEUE:-$FAMILY_PLAN_DIR/adaptive_cn_fill.csv}"
ADAPTIVE_REGEN_REPORTS_DIR="${ADAPTIVE_REGEN_REPORTS_DIR:-$FAMILY_REPORTS_ROOT/adaptive_cn_fill}"
ADAPTIVE_REGEN_OUT="${ADAPTIVE_REGEN_OUT:-$FAMILY_OUT_ROOT/adaptive_cn_fill}"
ADAPTIVE_REGEN_NUM_SHARDS="${ADAPTIVE_REGEN_NUM_SHARDS:-16}"
ADAPTIVE_REGEN_MAX_CONCURRENT_TASKS="${ADAPTIVE_REGEN_MAX_CONCURRENT_TASKS:-4}"
ADAPTIVE_REGEN_MAX_ATTEMPTS="${ADAPTIVE_REGEN_MAX_ATTEMPTS:-2}"
ADAPTIVE_REGEN_TIMEOUT_PER_COMPLEX="${ADAPTIVE_REGEN_TIMEOUT_PER_COMPLEX:-3600}"
ADAPTIVE_REGEN_PROFILE_SEQUENCE="${ADAPTIVE_REGEN_PROFILE_SEQUENCE:-standard:large_ligand_fast}"
ADAPTIVE_REGEN_MEM="${ADAPTIVE_REGEN_MEM:-3G}"

MISSING_BUILD_EXPECTED_ROWS="${MISSING_BUILD_EXPECTED_ROWS:-23}"
MISSING_BUILD_MAX_ROWS="${MISSING_BUILD_MAX_ROWS:-23}"
MISSING_BUILD_NUM_SHARDS="${MISSING_BUILD_NUM_SHARDS:-23}"
MISSING_BUILD_MAX_CONCURRENT_TASKS="${MISSING_BUILD_MAX_CONCURRENT_TASKS:-6}"
MISSING_BUILD_MAX_ATTEMPTS="${MISSING_BUILD_MAX_ATTEMPTS:-2}"
MISSING_BUILD_TIMEOUT_PER_COMPLEX="${MISSING_BUILD_TIMEOUT_PER_COMPLEX:-3000}"
MISSING_BUILD_MAX_TOTAL_SECONDS="${MISSING_BUILD_MAX_TOTAL_SECONDS:-20400}"
MISSING_BUILD_SEED="${MISSING_BUILD_SEED:-104729}"
MISSING_BUILD_SEED_STEP="${MISSING_BUILD_SEED_STEP:-15485863}"
MISSING_BUILD_PROFILE_SEQUENCE="${MISSING_BUILD_PROFILE_SEQUENCE:-large_ligand_fast:standard:emergency_unrelaxed:large_ligand_deep}"
MISSING_BUILD_WALLTIME="${MISSING_BUILD_WALLTIME:-06:00:00}"
MISSING_BUILD_MEM="${MISSING_BUILD_MEM:-3G}"
MISSING_BUILD_QUEUE="${MISSING_BUILD_QUEUE:-reports/missing_geometry_build_queue_$(date -u +%Y%m%dT%H%M%SZ).csv}"

# Lightweight coordinator submitted after the current adaptive merge.  It
# prepares immutable queues only after that parent finishes, then submits the
# 12-row canonical build followed by the ten non-pincer family groups.
FOLLOWUP_COORD_WALLTIME="${FOLLOWUP_COORD_WALLTIME:-00:30:00}"
FOLLOWUP_COORD_MEM="${FOLLOWUP_COORD_MEM:-2G}"
FOLLOWUP_CHAIN_ROOT="${FOLLOWUP_CHAIN_ROOT:-reports/gradual_followup_chains}"
FOLLOWUP_EXPECTED_ADAPTIVE_ROWS="${FOLLOWUP_EXPECTED_ADAPTIVE_ROWS:-20}"
FOLLOWUP_EXPECTED_ADAPTIVE_EXCLUDED="${FOLLOWUP_EXPECTED_ADAPTIVE_EXCLUDED:-7}"
FOLLOWUP_EXPECTED_ADAPTIVE_STILL_FAILED="${FOLLOWUP_EXPECTED_ADAPTIVE_STILL_FAILED:-3}"
FOLLOWUP_EXPECTED_FAMILY_ROWS="${FOLLOWUP_EXPECTED_FAMILY_ROWS:-86}"
FOLLOWUP_EXPECTED_FAILED_QC_DEFERRED="${FOLLOWUP_EXPECTED_FAILED_QC_DEFERRED:-10}"
FOLLOWUP_PINCER_FAILED_INPUT="${FOLLOWUP_PINCER_FAILED_INPUT:-$FAMILY_REPORTS_ROOT/template_pincer/regenerated_fail_long_bond_still_failed.csv}"
FOLLOWUP_FAMILY_GROUPS=(
  template_flavonol
  template_aminopolycarboxylate
  template_amide_core
  template_large_rigid
  template_other
  conformer_borderline
  conformer_sibling
  conformer_deep
  no_xyz_uff
  no_xyz_ligtype
)

# Focused rescue of the ten failures from job 5981236.  Each phase is a
# five-task array (one molecule per task); phase two waits for phase one.
HARD10_NUM_SHARDS="${HARD10_NUM_SHARDS:-5}"
HARD10_MAX_CONCURRENT_TASKS="${HARD10_MAX_CONCURRENT_TASKS:-5}"
HARD10_CPUS="${HARD10_CPUS:-4}"
HARD10_MEM="${HARD10_MEM:-5G}"
HARD10_WALLTIME="${HARD10_WALLTIME:-12:00:00}"
HARD10_MAX_ATTEMPTS="${HARD10_MAX_ATTEMPTS:-2}"
HARD10_TIMEOUT_PER_COMPLEX="${HARD10_TIMEOUT_PER_COMPLEX:-7200}"
HARD10_MAX_TOTAL_SECONDS="${HARD10_MAX_TOTAL_SECONDS:-39600}"
HARD10_SEED="${HARD10_SEED:-330287}"
HARD10_SEED_STEP="${HARD10_SEED_STEP:-15485863}"
HARD10_NO_STRUCTURES_PROFILE_SEQUENCE="${HARD10_NO_STRUCTURES_PROFILE_SEQUENCE:-uff_xtb_no_preopt:uff_unrelaxed}"
HARD10_NATIVE_CRASH_PROFILE_SEQUENCE="${HARD10_NATIVE_CRASH_PROFILE_SEQUENCE:-uff_xtb_no_preopt:uff_unrelaxed}"
HARD10_NO_STRUCTURES_QUEUE="${HARD10_NO_STRUCTURES_QUEUE:-reports/hard10_no_structures_queue.csv}"
HARD10_NATIVE_CRASH_QUEUE="${HARD10_NATIVE_CRASH_QUEUE:-reports/hard10_native_crash_queue.csv}"
HARD10_FOLLOWUP_REGEN="${HARD10_FOLLOWUP_REGEN:-1}"

# Optional parent for a standalone submit-regeneration call.  When set, the
# regeneration array is submitted now but remains pending until this job ends.
REGEN_START_AFTER_JOB="${REGEN_START_AFTER_JOB:-}"

DRY_RUN="${DRY_RUN:-0}"
ALLOW_OVERLAP="${ALLOW_OVERLAP:-0}"
QOS_MAX_WALLTIME="${QOS_MAX_WALLTIME:-24:00:00}"
QOS_MAX_SUBMIT_JOBS="${QOS_MAX_SUBMIT_JOBS:-96}"
QOS_MAX_RUNNING_JOBS="${QOS_MAX_RUNNING_JOBS:-48}"

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" != /* ]]; then
  SCRIPT_PATH="$(pwd)/$SCRIPT_PATH"
fi

CREATED_JOBS=()
CURRENT_STAGE="initialization"
SUBMITTED_JOB_ID=""
DRY_JOB_COUNTER=0
LAST_REGEN_JOB_ID=""
LAST_MERGE_REGEN_JOB_ID=""
LAST_BUILD_JOB_ID=""
LAST_POST_QC_JOB_ID=""

usage() {
  cat <<'EOF'
Usage:
  bash slurm/geometry_pipeline.slurm submit
  bash slurm/geometry_pipeline.slurm submit-remaining
  bash slurm/geometry_pipeline.slurm submit-ligtype-rescue
  bash slurm/geometry_pipeline.slurm submit-regeneration
  bash slurm/geometry_pipeline.slurm submit-missing
  bash slurm/geometry_pipeline.slurm submit-regeneration-no-candidate
  bash slurm/geometry_pipeline.slurm prepare-family-regeneration
  bash slurm/geometry_pipeline.slurm family-regeneration-status
  bash slurm/geometry_pipeline.slurm submit-family-regeneration [group]
  bash slurm/geometry_pipeline.slurm submit-family-regeneration-chain [group ...]
  bash slurm/geometry_pipeline.slurm submit-adaptive-regeneration
  bash slurm/geometry_pipeline.slurm submit-all-remaining
  bash slurm/geometry_pipeline.slurm all-remaining-status
  bash slurm/geometry_pipeline.slurm submit-gradual-followup [adaptive-merge-job-id]
  bash slurm/geometry_pipeline.slurm submit-hard10-rescue
  DRY_RUN=1 bash slurm/geometry_pipeline.slurm submit

Important env knobs:
  NUM_SHARDS=16
  MAX_CONCURRENT_TASKS=8
  REGEN_NUM_SHARDS=36
  REGEN_MAX_CONCURRENT_TASKS=6
  BUILD_WALLTIME=09:00:00
  REGEN_WALLTIME=12:00:00
  BUILD_CPUS=4
  BUILD_MEM=2G
  REGEN_CPUS=4
  REGEN_MEM=2G
  BUILD_MAX_ATTEMPTS=3
  BUILD_TIMEOUT_PER_COMPLEX=7200
  REGEN_MAX_ATTEMPTS=2
  REGEN_TIMEOUT_PER_COMPLEX=2400
  REGEN_WALLTIME_MARGIN_SECONDS=1800
  REGEN_SEED=61453
  REGEN_PROFILE_SEQUENCE=standard
  LIGTYPE_OVERRIDES=data/processed/ligtype_overrides.csv
  LIGTYPE_RESCUE_OVERRIDES=reports/geometry_ligtype_rescue_queue.csv
  RETRY_STATUS_LIST=skipped_known_bad_ligtype,ok_simplified_ligand
  RETRY_ONLY=1

Hard10 rescue defaults:
  5 shards, 5 simultaneous tasks, 4 CPUs and 5G per task
  second five-task array starts afterany the first array
  FAIL_LONG_BOND regeneration starts afterany the second array
  safe alongside the existing 21/22-shard ln_geom tasks (isolated queues/indexes/logs)
  HARD10_FOLLOWUP_REGEN=0 disables the regeneration follow-up

Standalone dependency example:
  REGEN_START_AFTER_JOB=123456 bash slurm/geometry_pipeline.slurm submit-regeneration

Family regeneration groups (submit one bounded group at a time):
  template_pincer, template_aminopolycarboxylate, template_amide_core,
  template_large_rigid, template_flavonol, template_other, conformer_borderline,
  conformer_sibling, conformer_deep, no_xyz_uff, no_xyz_ligtype

Pilot a representative subset while keeping the same resumable report/output:
  FAMILY_MAX_ROWS=8 bash slurm/geometry_pipeline.slurm submit-family-regeneration template_pincer

Submit every non-empty family as a sequential array -> merge dependency chain:
  bash slurm/geometry_pipeline.slurm submit-family-regeneration-chain

Prepare and submit the bounded adjacent-CN/fill hypotheses diagnosed from the
completed template_pincer failures:
  bash slurm/geometry_pipeline.slurm submit-adaptive-regeneration

Audit the current cluster checkout, freeze every unresolved source into
non-overlapping immutable queues, and submit a sequential route chain plus a
strict final status job:
  DRY_RUN=1 bash slurm/geometry_pipeline.slurm submit-all-remaining
  bash slurm/geometry_pipeline.slurm submit-all-remaining

The default scope is known-unfinished. To explicitly re-audit every historical
success as well (a much broader plan), opt in with:
  ALL_REMAINING_SCOPE=strict-baseline-audit DRY_RUN=1 \
    bash slurm/geometry_pipeline.slurm submit-all-remaining

Resume the exact same plan after an interruption (do not edit its queues):
  ALL_REMAINING_PLAN_ID=<printed-plan-id> ALL_REMAINING_PLAN_PREPARED=1 \
    bash slurm/geometry_pipeline.slurm submit-all-remaining

After the adaptive merge finishes, prepare fresh immutable queues, generate
the 12 missing canonical XYZ files, then run the 86 non-pincer family rows:
  bash slurm/geometry_pipeline.slurm submit-gradual-followup

To queue the coordinator before an adaptive merge finishes, pass its job ID:
  bash slurm/geometry_pipeline.slurm submit-gradual-followup 6014302

This file is also the SLURM worker. Do not call worker steps manually unless you
are debugging:
  PIPELINE_STEP=build sbatch slurm/geometry_pipeline.slurm
EOF
}

walltime_to_seconds() {
  local value="$1"
  local days=0
  local rest="$value"
  if [[ "$value" == *-* ]]; then
    days="${value%%-*}"
    rest="${value#*-}"
  fi
  local hours minutes seconds
  IFS=: read -r hours minutes seconds <<< "$rest"
  if [ -z "${seconds:-}" ]; then
    echo "Invalid walltime format: $value" >&2
    return 2
  fi
  echo $((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
}

slurm_user() {
  local user_name="${SLURM_JOB_USER:-${USER:-}}"
  if [ -z "$user_name" ]; then
    user_name="$(id -un)"
  fi
  echo "$user_name"
}

setup_python() {
  if [ ! -x "$PY" ]; then
    if command -v conda >/dev/null 2>&1; then
      local conda_base
      conda_base="$(conda info --base 2>/dev/null || true)"
      if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate arch
        PY="$(command -v python)"
      fi
    fi
  fi
  if [ ! -x "$PY" ]; then
    echo "Missing arch Python: $PY" >&2
    exit 2
  fi

  export PATH="$(dirname "$PY"):$PATH"
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
}

check_full_stack() {
  "$PY" -V
  "$PY" -c "import sys; print(sys.executable)"
  "$PY" - <<'PY'
from architector import build_complex
print("architector build_complex OK")
import pandas
print("pandas OK")
from rdkit import Chem
print("rdkit OK")
try:
    import xtb
    print("xtb python package OK")
except Exception as exc:
    print(f"xtb python package unavailable: {exc}")
try:
    from openbabel import openbabel
    print("openbabel OK")
except Exception as exc:
    raise SystemExit(f"openbabel import failed: {exc}")
PY
}

check_pandas_only() {
  "$PY" -V
  "$PY" -c "import pandas; print('pandas OK')"
}

ensure_repo_dirs() {
  mkdir -p logs reports data/geometries data/processed data/processed/audit
  mkdir -p data/processed/geometries_regenerated_fail_long_bond
}

require_geometry_specs() {
  if [ ! -f data/processed/geometry_specs.csv ]; then
    echo "Missing data/processed/geometry_specs.csv. Run scripts/build_dataset_no3d.py before submitting." >&2
    exit 3
  fi
}

print_cancel_hint() {
  if [ "${#CREATED_JOBS[@]}" -gt 0 ]; then
    echo "Safe cancel command for jobs submitted by this script:"
    echo "  scancel ${CREATED_JOBS[*]}"
  fi
}

on_submit_error() {
  local exit_code=$?
  echo >&2
  echo "ERROR: submission failed at stage: ${CURRENT_STAGE}" >&2
  echo "Exit code: ${exit_code}" >&2
  if [ "${#CREATED_JOBS[@]}" -gt 0 ]; then
    echo "Already submitted job IDs: ${CREATED_JOBS[*]}" >&2
    echo "Safe cancel command:" >&2
    echo "  scancel ${CREATED_JOBS[*]}" >&2
  else
    echo "No jobs were submitted by this script before the failure." >&2
  fi
  exit "$exit_code"
}

submit_sbatch() {
  local label="$1"
  shift
  CURRENT_STAGE="$label"
  if [ "$DRY_RUN" = "1" ]; then
    DRY_JOB_COUNTER=$((DRY_JOB_COUNTER + 1))
    SUBMITTED_JOB_ID="DRY${DRY_JOB_COUNTER}"
    printf 'DRY RUN [%s]: sbatch' "$label"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi

  local raw job
  raw="$(sbatch --parsable "$@")"
  job="${raw%%;*}"
  if [ -z "$job" ] || ! [[ "$job" =~ ^[0-9]+$ ]]; then
    echo "sbatch returned an invalid job id for ${label}: ${raw}" >&2
    return 11
  fi
  CREATED_JOBS+=("$job")
  SUBMITTED_JOB_ID="$job"
}

require_sbatch_if_needed() {
  if [ "$DRY_RUN" != "1" ] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available. Run this on the cluster login node or use DRY_RUN=1." >&2
    exit 2
  fi
}

check_submit_limits() {
  local planned_jobs="$1"
  local current_submitted="unknown"
  local current_running="unknown"
  if command -v squeue >/dev/null 2>&1; then
    local user_name
    user_name="$(slurm_user)"
    current_submitted="$(squeue -h -r -u "$user_name" | wc -l | tr -d ' ')"
    current_running="$(squeue -h -r -u "$user_name" -t R | wc -l | tr -d ' ')"
    if (( current_submitted + planned_jobs > QOS_MAX_SUBMIT_JOBS )); then
      echo "Current submitted jobs plus planned jobs exceed QOS_MAX_SUBMIT_JOBS:" >&2
      echo "  current=$current_submitted planned=$planned_jobs limit=$QOS_MAX_SUBMIT_JOBS" >&2
      exit 9
    fi
  fi
  echo "CURRENT_SUBMITTED_JOBS=$current_submitted"
  echo "CURRENT_RUNNING_JOBS=$current_running"
  echo "PLANNED_SUBMITTED_JOBS=$planned_jobs"
}

check_no_overlap() {
  if [ "$ALLOW_OVERLAP" = "1" ] || ! command -v squeue >/dev/null 2>&1; then
    return 0
  fi
  local user_name active
  user_name="$(slurm_user)"
  active="$(squeue -h -u "$user_name" -o "%i %j %T" | awk -v current="${SLURM_JOB_ID:-}" '$1 != current && $2 ~ /^ln_(geom|post_qc|regen_bad3d|merge_regen)/ {print}')"
  if [ -n "$active" ]; then
    echo "Active geometry pipeline jobs already exist:" >&2
    echo "$active" >&2
    echo "Refusing to submit overlapping chains because they compete for priority and can collide on reports." >&2
    echo "Set ALLOW_OVERLAP=1 only if you are intentionally debugging separate outputs." >&2
    exit 12
  fi
}

check_no_hard10_overlap() {
  if [ "$ALLOW_OVERLAP" = "1" ] || ! command -v squeue >/dev/null 2>&1; then
    return 0
  fi
  local user_name active
  user_name="$(slurm_user)"
  active="$(squeue -h -u "$user_name" -o "%i %j %T" | awk -v current="${SLURM_JOB_ID:-}" '$1 != current && $2 ~ /^ln_hard10_(no_struct|native)/ {print}')"
  if [ -n "$active" ]; then
    echo "An active hard10 rescue chain already exists:" >&2
    echo "$active" >&2
    echo "Refusing a duplicate hard10 submission. The ordinary ln_geom array may keep running." >&2
    exit 12
  fi
}

check_no_regen_overlap() {
  if [ "$ALLOW_OVERLAP" = "1" ] || ! command -v squeue >/dev/null 2>&1; then
    return 0
  fi
  local user_name active
  user_name="$(slurm_user)"
  active="$(squeue -h -u "$user_name" -o "%i %j %T" | awk -v current="${SLURM_JOB_ID:-}" '$1 != current && $2 ~ /^ln_(regen_bad3d|merge_regen)/ {print}')"
  if [ -n "$active" ]; then
    echo "An active FAIL_LONG_BOND regeneration chain already exists:" >&2
    echo "$active" >&2
    echo "Refusing a second chain because regeneration reports and outputs would collide." >&2
    exit 12
  fi
}

check_no_followup_overlap() {
  if [ "$ALLOW_OVERLAP" = "1" ] || ! command -v squeue >/dev/null 2>&1; then
    return 0
  fi
  local user_name active
  user_name="$(slurm_user)"
  active="$(squeue -h -u "$user_name" -o "%i %j %T" | awk -v current="${SLURM_JOB_ID:-}" '$1 != current && $2 == "ln_geom_followup" {print}')"
  if [ -n "$active" ]; then
    echo "An adaptive follow-up coordinator is already active:" >&2
    echo "$active" >&2
    echo "Refusing to queue a duplicate gradual follow-up." >&2
    exit 12
  fi
}

validate_hard10_config() {
  local name value
  for name in HARD10_NUM_SHARDS HARD10_MAX_CONCURRENT_TASKS HARD10_CPUS HARD10_MAX_ATTEMPTS HARD10_TIMEOUT_PER_COMPLEX HARD10_MAX_TOTAL_SECONDS; do
    value="${!name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
      echo "$name must be a non-negative integer, got: $value" >&2
      exit 2
    fi
  done
  if (( HARD10_NUM_SHARDS != 5 )); then
    echo "HARD10_NUM_SHARDS must remain 5: each rescue phase contains exactly five molecules." >&2
    exit 2
  fi
  if (( HARD10_MAX_CONCURRENT_TASKS != 5 )); then
    echo "HARD10_MAX_CONCURRENT_TASKS must remain 5 so all five molecules run simultaneously." >&2
    exit 2
  fi
  if (( HARD10_CPUS < 1 || HARD10_MAX_ATTEMPTS < 1 || HARD10_TIMEOUT_PER_COMPLEX < 1 )); then
    echo "HARD10_CPUS, HARD10_MAX_ATTEMPTS, and HARD10_TIMEOUT_PER_COMPLEX must be >= 1." >&2
    exit 2
  fi
  if (( HARD10_MAX_CONCURRENT_TASKS > QOS_MAX_RUNNING_JOBS )); then
    echo "Hard10 concurrency exceeds QOS_MAX_RUNNING_JOBS=$QOS_MAX_RUNNING_JOBS." >&2
    exit 7
  fi
  local hard10_wall_seconds qos_wall_seconds
  hard10_wall_seconds="$(walltime_to_seconds "$HARD10_WALLTIME")"
  qos_wall_seconds="$(walltime_to_seconds "$QOS_MAX_WALLTIME")"
  if (( hard10_wall_seconds > qos_wall_seconds )); then
    echo "HARD10_WALLTIME=$HARD10_WALLTIME exceeds QOS_MAX_WALLTIME=$QOS_MAX_WALLTIME." >&2
    exit 5
  fi
  if (( HARD10_MAX_TOTAL_SECONDS >= hard10_wall_seconds )); then
    echo "HARD10_MAX_TOTAL_SECONDS must leave time before the SLURM walltime." >&2
    exit 5
  fi
  case "$HARD10_FOLLOWUP_REGEN" in
    0|1) ;;
    *) echo "HARD10_FOLLOWUP_REGEN must be 0 or 1, got: $HARD10_FOLLOWUP_REGEN" >&2; exit 2 ;;
  esac
}

prepare_hard10_queues() {
  setup_python
  check_pandas_only
  require_geometry_specs
  if [ ! -f "$LIGTYPE_OVERRIDES" ]; then
    echo "Missing ligType overrides: $LIGTYPE_OVERRIDES" >&2
    exit 3
  fi
  "$PY" scripts/build_unique_geometries.py prepare-hard10-rescue \
    --specs data/processed/geometry_specs.csv \
    --ligtype-overrides "$LIGTYPE_OVERRIDES" \
    --hard10-no-structures-queue-output "$HARD10_NO_STRUCTURES_QUEUE" \
    --hard10-native-crash-queue-output "$HARD10_NATIVE_CRASH_QUEUE"
}

validate_common_config() {
  if (( NUM_SHARDS < 1 )); then
    echo "NUM_SHARDS must be >= 1" >&2
    exit 2
  fi
  if (( MAX_CONCURRENT_TASKS < 1 || MAX_CONCURRENT_TASKS > NUM_SHARDS )); then
    echo "MAX_CONCURRENT_TASKS must be in [1, NUM_SHARDS]" >&2
    exit 2
  fi
  if (( MAX_CONCURRENT_TASKS > QOS_MAX_RUNNING_JOBS )); then
    echo "MAX_CONCURRENT_TASKS=$MAX_CONCURRENT_TASKS exceeds QOS_MAX_RUNNING_JOBS=$QOS_MAX_RUNNING_JOBS" >&2
    exit 7
  fi
}

validate_regen_config() {
  local name value
  for name in \
    REGEN_NUM_SHARDS REGEN_MAX_CONCURRENT_TASKS REGEN_MAX_ATTEMPTS \
    REGEN_TIMEOUT_PER_COMPLEX REGEN_WALLTIME_MARGIN_SECONDS \
    REGEN_N_SYMMETRIES REGEN_N_SYMMETRIES_STEP REGEN_N_CONFORMERS \
    REGEN_N_CONFORMERS_STEP REGEN_XTB_MAX_ITERATIONS; do
    value="${!name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
      echo "$name must be a non-negative integer, got: $value" >&2
      exit 2
    fi
  done
  if (( REGEN_NUM_SHARDS < 1 )); then
    echo "REGEN_NUM_SHARDS must be >= 1" >&2
    exit 2
  fi
  if (( REGEN_MAX_CONCURRENT_TASKS < 1 || REGEN_MAX_CONCURRENT_TASKS > REGEN_NUM_SHARDS )); then
    echo "REGEN_MAX_CONCURRENT_TASKS must be in [1, REGEN_NUM_SHARDS]" >&2
    exit 2
  fi
  if (( REGEN_MAX_CONCURRENT_TASKS > QOS_MAX_RUNNING_JOBS )); then
    echo "REGEN_MAX_CONCURRENT_TASKS=$REGEN_MAX_CONCURRENT_TASKS exceeds QOS_MAX_RUNNING_JOBS=$QOS_MAX_RUNNING_JOBS" >&2
    exit 7
  fi
  if (( REGEN_MAX_ATTEMPTS < 1 || REGEN_TIMEOUT_PER_COMPLEX < 1 )); then
    echo "REGEN_MAX_ATTEMPTS and REGEN_TIMEOUT_PER_COMPLEX must be >= 1" >&2
    exit 2
  fi
  if (( REGEN_N_SYMMETRIES < 1 || REGEN_N_CONFORMERS < 1 || REGEN_XTB_MAX_ITERATIONS < 1 )); then
    echo "REGEN_N_SYMMETRIES, REGEN_N_CONFORMERS, and REGEN_XTB_MAX_ITERATIONS must be >= 1" >&2
    exit 2
  fi
}

validate_regen_walltime() {
  local regen_wall_seconds qos_wall_seconds
  regen_wall_seconds="$(walltime_to_seconds "$REGEN_WALLTIME")"
  qos_wall_seconds="$(walltime_to_seconds "$QOS_MAX_WALLTIME")"
  if (( regen_wall_seconds > qos_wall_seconds )); then
    echo "REGEN_WALLTIME=$REGEN_WALLTIME exceeds QOS_MAX_WALLTIME=$QOS_MAX_WALLTIME." >&2
    echo "Use REGEN_WALLTIME=24:00:00 or lower on campus." >&2
    exit 5
  fi
}

submit_full_chain() {
  trap on_submit_error ERR
  require_sbatch_if_needed
  validate_common_config
  validate_regen_walltime
  if [ "$SUBMIT_REGEN_AFTER_QC" = "1" ]; then
    validate_regen_config
  fi
  case "$POST_BUILD_DEPENDENCY" in
    afterok|afterany) ;;
    *) echo "POST_BUILD_DEPENDENCY must be afterok or afterany, got $POST_BUILD_DEPENDENCY" >&2; exit 2 ;;
  esac
  ensure_repo_dirs
  check_no_overlap
  check_submit_limits "$((NUM_SHARDS + 1))"

  local array_spec
  array_spec="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT_TASKS}"

  if [ "$SUBMIT_REGEN_AFTER_QC" = "1" ]; then
    echo "Submitting geometry build + QC + delayed regeneration chain"
  else
    echo "Submitting geometry build + QC chain"
  fi
  echo "PROJECT_DIR=$PROJECT_DIR"
  echo "NUM_SHARDS=$NUM_SHARDS"
  echo "MAX_CONCURRENT_TASKS=$MAX_CONCURRENT_TASKS"
  echo "ARRAY_SPEC=$array_spec"
  echo "BUILD_WALLTIME=$BUILD_WALLTIME"
  echo "POST_QC_WALLTIME=$POST_QC_WALLTIME"
  echo "REGEN_NUM_SHARDS=$REGEN_NUM_SHARDS"
  echo "REGEN_MAX_CONCURRENT_TASKS=$REGEN_MAX_CONCURRENT_TASKS"
  echo "REGEN_WALLTIME=$REGEN_WALLTIME"
  echo "MERGE_REGEN_WALLTIME=$MERGE_REGEN_WALLTIME"
  echo "BUILD_MAX_ATTEMPTS=$BUILD_MAX_ATTEMPTS"
  echo "BUILD_TIMEOUT_PER_COMPLEX=$BUILD_TIMEOUT_PER_COMPLEX"
  echo "MAX_TOTAL_SECONDS=$MAX_TOTAL_SECONDS"
  echo "SEED=$SEED"
  echo "SEED_STEP=$SEED_STEP"
  echo "PROFILE_SEQUENCE=$PROFILE_SEQUENCE"
  echo "SPECS_CSV=$SPECS_CSV"
  echo "FIXED_MISSING_QUEUE=$FIXED_MISSING_QUEUE"
  echo "INDEX_TAG=$INDEX_TAG"
  echo "REGEN_MAX_ATTEMPTS=$REGEN_MAX_ATTEMPTS"
  echo "REGEN_TIMEOUT_PER_COMPLEX=$REGEN_TIMEOUT_PER_COMPLEX"
  echo "RETRY_FAILED=$RETRY_FAILED"
  echo "RETRY_STATUS_LIST=$RETRY_STATUS_LIST"
  echo "RETRY_ONLY=$RETRY_ONLY"
  echo "MISSING_ONLY=$MISSING_ONLY"
  echo "HARD_LIGAND_MODE=$HARD_LIGAND_MODE"
  echo "POST_BUILD_DEPENDENCY=$POST_BUILD_DEPENDENCY"
  echo "VERIFY_MISSING_COMPLETE=$VERIFY_MISSING_COMPLETE"
  echo "LIGTYPE_OVERRIDES=$LIGTYPE_OVERRIDES"
  echo "DRY_RUN=$DRY_RUN"
  echo

  submit_sbatch "build array" \
    --job-name=ln_geom \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --array="$array_spec" \
    --cpus-per-task="$BUILD_CPUS" \
    --mem="$BUILD_MEM" \
    --time="$BUILD_WALLTIME" \
    --chdir="$PROJECT_DIR" \
    --output=logs/ln_geom_%A_%a.out \
    --error=logs/ln_geom_%A_%a.err \
    --export=ALL,PIPELINE_STEP=build,NUM_SHARDS="$NUM_SHARDS",MAX_ATTEMPTS="$BUILD_MAX_ATTEMPTS",TIMEOUT_PER_COMPLEX="$BUILD_TIMEOUT_PER_COMPLEX",MAX_TOTAL_SECONDS="$MAX_TOTAL_SECONDS",SEED="$SEED",SEED_STEP="$SEED_STEP",PROFILE_SEQUENCE="$PROFILE_SEQUENCE",SPECS_CSV="$SPECS_CSV",FIXED_MISSING_QUEUE="$FIXED_MISSING_QUEUE",RETRY_FAILED="$RETRY_FAILED",RETRY_STATUS_LIST="$RETRY_STATUS_LIST",RETRY_ONLY="$RETRY_ONLY",MISSING_ONLY="$MISSING_ONLY",HARD_LIGAND_MODE="$HARD_LIGAND_MODE",LIGTYPE_OVERRIDES="$LIGTYPE_OVERRIDES" \
    "$SCRIPT_PATH" worker "$@"
  local build_job="$SUBMITTED_JOB_ID"
  LAST_BUILD_JOB_ID="$build_job"
  echo "build array job: $build_job"

  submit_sbatch "post-build QC" \
    --job-name=ln_post_qc \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --cpus-per-task="$POST_QC_CPUS" \
    --mem="$POST_QC_MEM" \
    --time="$POST_QC_WALLTIME" \
    --chdir="$PROJECT_DIR" \
    --output=logs/ln_post_qc_%j.out \
    --error=logs/ln_post_qc_%j.err \
    --dependency="$POST_BUILD_DEPENDENCY":"$build_job" \
    --export=ALL,PIPELINE_STEP=post_qc,NUM_SHARDS="$NUM_SHARDS",SUBMIT_REGEN_AFTER_QC="$SUBMIT_REGEN_AFTER_QC",VERIFY_MISSING_COMPLETE="$VERIFY_MISSING_COMPLETE",MISSING_BUILD_EXPECTED_ROWS="$MISSING_BUILD_EXPECTED_ROWS",REGEN_NUM_SHARDS="$REGEN_NUM_SHARDS",REGEN_MAX_CONCURRENT_TASKS="$REGEN_MAX_CONCURRENT_TASKS",REGEN_WALLTIME="$REGEN_WALLTIME",MERGE_REGEN_WALLTIME="$MERGE_REGEN_WALLTIME",REGEN_MAX_ATTEMPTS="$REGEN_MAX_ATTEMPTS",REGEN_TIMEOUT_PER_COMPLEX="$REGEN_TIMEOUT_PER_COMPLEX",REGEN_WALLTIME_MARGIN_SECONDS="$REGEN_WALLTIME_MARGIN_SECONDS",REGEN_SEED="$REGEN_SEED",REGEN_SEED_STEP="$REGEN_SEED_STEP",REGEN_PROFILE_SEQUENCE="$REGEN_PROFILE_SEQUENCE",REGEN_REPORTS_DIR="$REGEN_REPORTS_DIR",REGEN_OUT="$REGEN_OUT",INPUT_CSV="$INPUT_CSV",EXPECTED_INPUT_ROWS="$EXPECTED_INPUT_ROWS",LIGTYPE_OVERRIDES="$LIGTYPE_OVERRIDES" \
    "$SCRIPT_PATH" worker
  local post_qc_job="$SUBMITTED_JOB_ID"
  LAST_POST_QC_JOB_ID="$post_qc_job"
  echo "post-build QC job: $post_qc_job"

  echo
  echo "Dependency chain:"
  echo "  build array                    $build_job"
  echo "  $POST_BUILD_DEPENDENCY(build) post-QC       $post_qc_job"
  if [ "$SUBMIT_REGEN_AFTER_QC" = "1" ]; then
    echo "  post-QC submits regen+merge only after QC queue is rebuilt"
  else
    echo "  post-QC does not submit regeneration in this mode"
  fi
  echo
  print_cancel_hint
  echo
  echo "Watch with:"
  echo "  squeue -u \"\$USER\""
}

submit_ligtype_rescue_chain() {
  RETRY_FAILED=0
  RETRY_STATUS_LIST="${RETRY_STATUS_LIST:-skipped_known_bad_ligtype}"
  RETRY_ONLY=1
  MISSING_ONLY=0
  HARD_LIGAND_MODE=0
  SUBMIT_REGEN_AFTER_QC=0

  if [ "$BUILD_WALLTIME" = "24:00:00" ]; then
    BUILD_WALLTIME="12:00:00"
  fi
  if [ "$BUILD_MAX_ATTEMPTS" = "3" ]; then
    BUILD_MAX_ATTEMPTS="2"
  fi
  if [ "$BUILD_TIMEOUT_PER_COMPLEX" = "7200" ]; then
    BUILD_TIMEOUT_PER_COMPLEX="3600"
  fi
  if [ "$MAX_CONCURRENT_TASKS" = "8" ]; then
    MAX_CONCURRENT_TASKS="4"
  fi

  if [ ! -f "$LIGTYPE_OVERRIDES" ] && [ -f "$LIGTYPE_RESCUE_OVERRIDES" ]; then
    LIGTYPE_OVERRIDES="$LIGTYPE_RESCUE_OVERRIDES"
  fi
  if [ ! -f "$LIGTYPE_OVERRIDES" ]; then
    echo "Missing ligType override file: $LIGTYPE_OVERRIDES" >&2
    echo "Create/copy data/processed/ligtype_overrides.csv or reports/geometry_ligtype_rescue_queue.csv before submitting rescue." >&2
    exit 4
  fi

  echo "Submitting targeted ligType rescue for skipped_known_bad_ligtype rows"
  submit_full_chain "$@"
}

submit_remaining_chain() {
  RETRY_FAILED=1
  RETRY_STATUS_LIST="${RETRY_STATUS_LIST:-skipped_known_bad_ligtype,ok_simplified_ligand}"
  RETRY_ONLY=1
  MISSING_ONLY=0
  HARD_LIGAND_MODE=0
  SUBMIT_REGEN_AFTER_QC=0

  if [ "$BUILD_MAX_ATTEMPTS" = "3" ]; then
    BUILD_MAX_ATTEMPTS="2"
  fi
  if [ "$BUILD_TIMEOUT_PER_COMPLEX" = "7200" ]; then
    BUILD_TIMEOUT_PER_COMPLEX="2400"
  fi
  if [ "$MAX_CONCURRENT_TASKS" = "8" ]; then
    MAX_CONCURRENT_TASKS="4"
  fi

  echo "Submitting targeted retry for remaining real-ligand geometry rows"
  submit_full_chain "$@"
}

submit_missing_build_chain() {
  setup_python
  require_geometry_specs
  if [ ! -f "$LIGTYPE_OVERRIDES" ]; then
    echo "Missing ligType override file required by missing-geometry build: $LIGTYPE_OVERRIDES" >&2
    exit 4
  fi

  local missing_rows
  missing_rows="$("$PY" scripts/build_unique_geometries.py count-missing-geometries)"
  if ! [[ "$missing_rows" =~ ^[0-9]+$ ]]; then
    echo "Could not determine missing canonical XYZ count: $missing_rows" >&2
    exit 2
  fi
  if (( missing_rows > MISSING_BUILD_MAX_ROWS )); then
    echo "Missing-geometry preflight found $missing_rows rows, above safety limit $MISSING_BUILD_MAX_ROWS." >&2
    echo "Raise MISSING_BUILD_MAX_ROWS only after checking the cluster files." >&2
    exit 9
  fi
  if [ "$missing_rows" = "0" ]; then
    echo "All canonical XYZ files already exist; nothing to submit."
    return 0
  fi
  if [ "$missing_rows" != "$MISSING_BUILD_EXPECTED_ROWS" ]; then
    echo "Expected exactly $MISSING_BUILD_EXPECTED_ROWS missing canonical XYZ files, found $missing_rows." >&2
    echo "Refusing to submit a stale or mismatched rescue queue." >&2
    exit 9
  fi

  echo "==== validate exact missing queue, ligTypes, and one-row shards ===="
  "$PY" scripts/build_unique_geometries.py plan-missing-geometries \
    --num-shards "$MISSING_BUILD_NUM_SHARDS" \
    --expected-missing-count "$MISSING_BUILD_EXPECTED_ROWS" \
    --ligtype-overrides "$LIGTYPE_OVERRIDES" \
    --missing-queue-output "$MISSING_BUILD_QUEUE"

  RETRY_FAILED=0
  RETRY_STATUS_LIST=""
  RETRY_ONLY=0
  MISSING_ONLY=1
  FIXED_MISSING_QUEUE=1
  HARD_LIGAND_MODE=0
  SUBMIT_REGEN_AFTER_QC=0
  POST_BUILD_DEPENDENCY=afterany
  VERIFY_MISSING_COMPLETE=1
  NUM_SHARDS="$MISSING_BUILD_NUM_SHARDS"
  MAX_CONCURRENT_TASKS="$MISSING_BUILD_MAX_CONCURRENT_TASKS"
  BUILD_MAX_ATTEMPTS="$MISSING_BUILD_MAX_ATTEMPTS"
  BUILD_TIMEOUT_PER_COMPLEX="$MISSING_BUILD_TIMEOUT_PER_COMPLEX"
  MAX_TOTAL_SECONDS="$MISSING_BUILD_MAX_TOTAL_SECONDS"
  SEED="$MISSING_BUILD_SEED"
  SEED_STEP="$MISSING_BUILD_SEED_STEP"
  PROFILE_SEQUENCE="$MISSING_BUILD_PROFILE_SEQUENCE"
  SPECS_CSV="$MISSING_BUILD_QUEUE"
  BUILD_WALLTIME="$MISSING_BUILD_WALLTIME"
  BUILD_MEM="$MISSING_BUILD_MEM"

  echo "Submitting normal stage-2 generation for $missing_rows missing canonical geometries"
  echo "Exactly one molecule per shard; at most $MAX_CONCURRENT_TASKS shards run together"
  echo "Frozen queue: $SPECS_CSV"
  echo "Outputs go directly to data/geometries and shard indexes"
  submit_full_chain "$@"
}

prepare_missing_regen_queue() {
  setup_python
  if [ ! -f "$MISSING_REGEN_LIGTYPE_OVERRIDES" ]; then
    echo "Missing focused ligType override file: $MISSING_REGEN_LIGTYPE_OVERRIDES" >&2
    exit 4
  fi
  "$PY" scripts/build_unique_geometries.py prepare-missing-regeneration \
    --input "$MISSING_REGEN_SOURCE_INPUT" \
    --still-failed-input "$MISSING_REGEN_STILL_FAILED" \
    --rescue-queue-output "$MISSING_REGEN_QUEUE"
}

submit_missing_regen_chain() {
  prepare_missing_regen_queue

  local missing_regen_rows
  missing_regen_rows="$("$PY" - "$MISSING_REGEN_QUEUE" <<'PY'
import sys
import pandas as pd
print(len(pd.read_csv(sys.argv[1], low_memory=False)))
PY
)"
  if [ "$missing_regen_rows" = "0" ]; then
    echo "No regeneration rows without XYZ remain; nothing to submit."
    return 0
  fi
  if [ -n "$MISSING_REGEN_EXPECTED_ROWS" ] && [ "$missing_regen_rows" != "$MISSING_REGEN_EXPECTED_ROWS" ]; then
    echo "Focused no-XYZ queue mismatch: got $missing_regen_rows, expected $MISSING_REGEN_EXPECTED_ROWS." >&2
    exit 10
  fi

  INPUT_CSV="$MISSING_REGEN_QUEUE"
  EXPECTED_INPUT_ROWS="$missing_regen_rows"
  LIGTYPE_OVERRIDES="$MISSING_REGEN_LIGTYPE_OVERRIDES"
  REGEN_REPORTS_DIR="$MISSING_REGEN_REPORTS_DIR"
  if (( MISSING_REGEN_NUM_SHARDS > missing_regen_rows )); then
    REGEN_NUM_SHARDS="$missing_regen_rows"
  else
    REGEN_NUM_SHARDS="$MISSING_REGEN_NUM_SHARDS"
  fi
  REGEN_MAX_CONCURRENT_TASKS="$MISSING_REGEN_MAX_CONCURRENT_TASKS"
  REGEN_MAX_ATTEMPTS="$MISSING_REGEN_MAX_ATTEMPTS"
  REGEN_TIMEOUT_PER_COMPLEX="$MISSING_REGEN_TIMEOUT_PER_COMPLEX"
  REGEN_SEED="$MISSING_REGEN_SEED"
  REGEN_SEED_STEP="$MISSING_REGEN_SEED_STEP"
  REGEN_PROFILE_SEQUENCE="$MISSING_REGEN_PROFILE_SEQUENCE"
  REGEN_MEM="$MISSING_REGEN_MEM"

  echo "Submitting focused rescue for $missing_regen_rows regeneration rows with no generated XYZ"
  submit_regen_chain "$@"
}

prepare_adaptive_regen_queue() {
  setup_python
  "$PY" scripts/build_unique_geometries.py prepare-adaptive-regeneration \
    --adaptive-input "$ADAPTIVE_REGEN_INPUT" \
    --adaptive-output "$ADAPTIVE_REGEN_QUEUE"
}

submit_adaptive_regen_chain() {
  prepare_adaptive_regen_queue

  local adaptive_rows
  adaptive_rows="$("$PY" - "$ADAPTIVE_REGEN_QUEUE" <<'PY'
import sys
import pandas as pd
print(len(pd.read_csv(sys.argv[1], low_memory=False)))
PY
)"
  if [ "$adaptive_rows" = "0" ]; then
    echo "No eligible adaptive CN/fill hypotheses remain; nothing to submit."
    return 0
  fi

  INPUT_CSV="$ADAPTIVE_REGEN_QUEUE"
  EXPECTED_INPUT_ROWS="$adaptive_rows"
  LIGTYPE_OVERRIDES="$FAMILY_LIGTYPE_OVERRIDES"
  REGEN_REPORTS_DIR="$ADAPTIVE_REGEN_REPORTS_DIR"
  REGEN_OUT="$ADAPTIVE_REGEN_OUT"
  if (( ADAPTIVE_REGEN_NUM_SHARDS > adaptive_rows )); then
    REGEN_NUM_SHARDS="$adaptive_rows"
  else
    REGEN_NUM_SHARDS="$ADAPTIVE_REGEN_NUM_SHARDS"
  fi
  if (( ADAPTIVE_REGEN_MAX_CONCURRENT_TASKS > REGEN_NUM_SHARDS )); then
    REGEN_MAX_CONCURRENT_TASKS="$REGEN_NUM_SHARDS"
  else
    REGEN_MAX_CONCURRENT_TASKS="$ADAPTIVE_REGEN_MAX_CONCURRENT_TASKS"
  fi
  REGEN_MAX_ATTEMPTS="$ADAPTIVE_REGEN_MAX_ATTEMPTS"
  REGEN_TIMEOUT_PER_COMPLEX="$ADAPTIVE_REGEN_TIMEOUT_PER_COMPLEX"
  REGEN_PROFILE_SEQUENCE="$ADAPTIVE_REGEN_PROFILE_SEQUENCE"
  REGEN_MEM="$ADAPTIVE_REGEN_MEM"

  echo "Submitting adaptive CN/fill regeneration for $adaptive_rows diagnosed hypotheses"
  echo "Source failures: $ADAPTIVE_REGEN_INPUT"
  echo "Queue: $INPUT_CSV"
  echo "Reports: $REGEN_REPORTS_DIR"
  echo "Outputs: $REGEN_OUT"
  submit_regen_chain "$@"
}

prepare_family_regen_plan() {
  setup_python
  "$PY" scripts/build_unique_geometries.py prepare-family-regeneration \
    --input "$MISSING_REGEN_SOURCE_INPUT" \
    --still-failed-input "$MISSING_REGEN_STILL_FAILED" \
    --reports-dir reports \
    --family-plan-dir "$FAMILY_PLAN_DIR"
}

family_regen_status() {
  setup_python
  "$PY" scripts/build_unique_geometries.py family-regeneration-status \
    --family-plan-dir "$FAMILY_PLAN_DIR" \
    --family-runs-dir "$FAMILY_REPORTS_ROOT"
}

submit_family_regen_group() {
  local group="${1:-$FAMILY_GROUP}"
  if [ "$#" -gt 0 ]; then
    shift
  fi
  if [ "$FAMILY_PLAN_PREPARED" != "1" ]; then
    prepare_family_regen_plan
  fi

  local config_file="$FAMILY_PLAN_DIR/run_config.csv"
  local config_line
  config_line="$("$PY" - "$config_file" "$group" <<'PY'
import sys
import pandas as pd

path, group = sys.argv[1:]
table = pd.read_csv(path, low_memory=False)
row = table[table["run_group"].astype(str) == group]
if row.empty:
    choices = ", ".join(table["run_group"].astype(str))
    raise SystemExit(f"Unknown FAMILY_GROUP={group!r}. Choose from: {choices}")
r = row.iloc[0]
fields = [
    "rows", "profile_sequence", "max_attempts", "timeout_per_complex",
    "num_shards", "max_concurrent_tasks", "memory",
]
print("|".join(str(r[name]) for name in fields))
PY
)"

  local rows profiles attempts timeout shards concurrency memory
  IFS='|' read -r rows profiles attempts timeout shards concurrency memory <<< "$config_line"
  if [ "$rows" = "0" ]; then
    echo "Family regeneration group $group is empty; nothing to submit."
    return 0
  fi

  INPUT_CSV="$FAMILY_PLAN_DIR/$group.csv"
  if [ -n "$FAMILY_MAX_ROWS" ]; then
    if ! [[ "$FAMILY_MAX_ROWS" =~ ^[1-9][0-9]*$ ]]; then
      echo "FAMILY_MAX_ROWS must be a positive integer, got: $FAMILY_MAX_ROWS" >&2
      exit 10
    fi
    if (( FAMILY_MAX_ROWS < rows )); then
      local pilot_queue="$FAMILY_PLAN_DIR/${group}_pilot${FAMILY_MAX_ROWS}.csv"
      "$PY" - "$INPUT_CSV" "$pilot_queue" "$FAMILY_MAX_ROWS" <<'PY'
import sys
import pandas as pd

source, output, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
table = pd.read_csv(source, low_memory=False)
strata = [
    name for name in ("ligand_family", "inner_sphere_anion")
    if name in table.columns
]
if strata:
    tie_breakers = [name for name in ("metal_symbol", "build_id") if name in table.columns]
    ordered = table.sort_values(strata + tie_breakers, kind="stable").copy()
    ordered["_pilot_round"] = ordered.groupby(strata, dropna=False).cumcount()
    pilot = ordered.sort_values(["_pilot_round"] + strata + tie_breakers, kind="stable").head(limit)
    pilot = pilot.drop(columns="_pilot_round")
else:
    pilot = table.sort_values("build_id", kind="stable").head(limit)
pilot.to_csv(output, index=False)
PY
      INPUT_CSV="$pilot_queue"
      rows="$FAMILY_MAX_ROWS"
      if (( shards > rows )); then
        shards="$rows"
      fi
      if (( concurrency > rows )); then
        concurrency="$rows"
      fi
      echo "Pilot mode: representative $rows-row subset -> $INPUT_CSV"
    fi
  fi
  EXPECTED_INPUT_ROWS="$rows"
  REGEN_REPORTS_DIR="$FAMILY_REPORTS_ROOT/$group"
  REGEN_OUT="$FAMILY_OUT_ROOT/$group"
  REGEN_NUM_SHARDS="$shards"
  REGEN_MAX_CONCURRENT_TASKS="$concurrency"
  REGEN_MAX_ATTEMPTS="$attempts"
  REGEN_TIMEOUT_PER_COMPLEX="$timeout"
  REGEN_PROFILE_SEQUENCE="$profiles"
  REGEN_MEM="$memory"
  if [[ "$group" == template_* ]]; then
    LIGTYPE_OVERRIDES="$FAMILY_LIGTYPE_OVERRIDES"
  elif [ "$group" = "no_xyz_ligtype" ]; then
    LIGTYPE_OVERRIDES="$MISSING_REGEN_LIGTYPE_OVERRIDES"
  else
    LIGTYPE_OVERRIDES="data/processed/ligtype_overrides.csv"
  fi

  echo "Submitting chemistry-family regeneration group: $group"
  echo "Queue: $INPUT_CSV ($rows rows)"
  echo "Profiles: $REGEN_PROFILE_SEQUENCE"
  echo "Attempts/time: $REGEN_MAX_ATTEMPTS x ${REGEN_TIMEOUT_PER_COMPLEX}s"
  echo "Shards/concurrency: $REGEN_NUM_SHARDS / $REGEN_MAX_CONCURRENT_TASKS"
  echo "Reports: $REGEN_REPORTS_DIR"
  echo "Outputs: $REGEN_OUT"
  submit_regen_chain "$@"
}

submit_family_regen_dependency_chain() {
  if [ -n "$FAMILY_MAX_ROWS" ]; then
    echo "FAMILY_MAX_ROWS is only supported for a single-family pilot, not a full dependency chain." >&2
    exit 10
  fi

  local -a groups=()
  if [ "$#" -gt 0 ]; then
    groups=("$@")
  else
    groups=(
      template_pincer
      template_flavonol
      template_aminopolycarboxylate
      template_amide_core
      template_large_rigid
      template_other
      conformer_borderline
      conformer_sibling
      conformer_deep
      no_xyz_uff
      no_xyz_ligtype
    )
  fi

  if [ "$FAMILY_PLAN_PREPARED" != "1" ]; then
    prepare_family_regen_plan
    FAMILY_PLAN_PREPARED=1
  fi
  FAMILY_DEPENDENCY_CHAIN=1

  local config_file="$FAMILY_PLAN_DIR/run_config.csv"
  local total_jobs
  total_jobs="$("$PY" - "$config_file" "${groups[@]}" <<'PY'
import sys
import pandas as pd

path, *groups = sys.argv[1:]
table = pd.read_csv(path, low_memory=False)
table["run_group"] = table["run_group"].astype(str)
known = set(table["run_group"])
unknown = [group for group in groups if group not in known]
if unknown:
    raise SystemExit(f"Unknown family groups: {', '.join(unknown)}")
by_group = table.set_index("run_group")
total = sum(
    int(float(by_group.loc[group, "num_shards"])) + 1
    for group in groups
    if int(float(by_group.loc[group, "rows"])) > 0
)
print(total)
PY
)"
  require_sbatch_if_needed
  check_submit_limits "$total_jobs"

  local next_dependency="${REGEN_START_AFTER_JOB:-}"
  local chain_stamp chain_file
  chain_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  chain_file="logs/family_regeneration_dependency_chain_${chain_stamp}.csv"
  local -a chain_rows=()
  if [ "$DRY_RUN" != "1" ]; then
    ensure_repo_dirs
    echo "run_group,start_after_job,regen_array_job,merge_job" > "$chain_file"
  fi

  local group dependency_label previous_merge
  for group in "${groups[@]}"; do
    dependency_label="${next_dependency:-none}"
    previous_merge="$LAST_MERGE_REGEN_JOB_ID"
    REGEN_START_AFTER_JOB="$next_dependency"
    REGEN_CHAIN_LABEL="family_${group}_${SLURM_JOB_ID:-manual}"

    echo
    echo "==== dependency-chain group: $group (after $dependency_label) ===="
    submit_family_regen_group "$group"
    if [ "$LAST_MERGE_REGEN_JOB_ID" = "$previous_merge" ]; then
      echo "No jobs created for empty group: $group"
      continue
    fi

    chain_rows+=("$group|$dependency_label|$LAST_REGEN_JOB_ID|$LAST_MERGE_REGEN_JOB_ID")
    if [ "$DRY_RUN" != "1" ]; then
      echo "$group,$dependency_label,$LAST_REGEN_JOB_ID,$LAST_MERGE_REGEN_JOB_ID" >> "$chain_file"
    fi
    next_dependency="$LAST_MERGE_REGEN_JOB_ID"
  done

  echo
  echo "Family regeneration dependency chain"
  for row in "${chain_rows[@]}"; do
    IFS='|' read -r group dependency_label regen_job merge_job <<< "$row"
    echo "  $group: after=$dependency_label array=$regen_job merge=$merge_job"
  done
  echo "Final merge job: ${next_dependency:-none}"
  if [ "$DRY_RUN" != "1" ]; then
    echo "Chain manifest: $chain_file"
    print_cancel_hint
  fi
}

resolve_all_remaining_dirs() {
  case "$ALL_REMAINING_SCOPE" in
    known-unfinished|strict-baseline-audit) ;;
    *)
      echo "ALL_REMAINING_SCOPE must be known-unfinished or strict-baseline-audit, got: $ALL_REMAINING_SCOPE" >&2
      exit 2
      ;;
  esac
  if [ -z "$ALL_REMAINING_PLAN_ID" ]; then
    if [ "$DRY_RUN" = "1" ] && [ "$ALL_REMAINING_PLAN_PREPARED" != "1" ]; then
      ALL_REMAINING_PLAN_ID="dryrun_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    else
      ALL_REMAINING_PLAN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    fi
  fi
  if ! [[ "$ALL_REMAINING_PLAN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ALL_REMAINING_PLAN_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
  fi

  if [ "$DRY_RUN" = "1" ] && [ "$ALL_REMAINING_PLAN_PREPARED" != "1" ]; then
    local dry_root
    dry_root="$(mktemp -d "${TMPDIR:-/tmp}/ln_all_remaining_dryrun.XXXXXX")"
    ALL_REMAINING_PLAN_DIR="${ALL_REMAINING_PLAN_DIR:-$dry_root/plan}"
    ALL_REMAINING_RUNS_DIR="${ALL_REMAINING_RUNS_DIR:-$dry_root/runs}"
    ALL_REMAINING_OUT_DIR="${ALL_REMAINING_OUT_DIR:-$dry_root/geometries}"
    echo "Dry-run plan workspace (temporary OS storage): $dry_root"
  else
    ALL_REMAINING_PLAN_DIR="${ALL_REMAINING_PLAN_DIR:-$ALL_REMAINING_PLAN_ROOT/$ALL_REMAINING_PLAN_ID}"
    ALL_REMAINING_RUNS_DIR="${ALL_REMAINING_RUNS_DIR:-$ALL_REMAINING_RUNS_ROOT/$ALL_REMAINING_PLAN_ID}"
    ALL_REMAINING_OUT_DIR="${ALL_REMAINING_OUT_DIR:-$ALL_REMAINING_OUT_ROOT/$ALL_REMAINING_PLAN_ID}"
  fi
}

prepare_all_remaining_plan() {
  setup_python
  check_pandas_only
  require_geometry_specs
  if [ "$ALL_REMAINING_PLAN_PREPARED" = "1" ]; then
    echo "Using pre-existing immutable all-remaining plan: $ALL_REMAINING_PLAN_DIR"
    return 0
  fi

  echo "==== audit and freeze all remaining source complexes ===="
  local plan_rc
  set +e
  "$PY" scripts/build_unique_geometries.py prepare-all-remaining \
    --specs "$SPECS_CSV" \
    --geometry-index data/processed/geometry_index_merged.csv \
    --historical-reports-dir reports \
    --family-plan-dir "$FAMILY_PLAN_DIR" \
    --family-runs-dir "$FAMILY_REPORTS_ROOT" \
    --adaptive-output "$ADAPTIVE_REGEN_QUEUE" \
    --ligtype-overrides "$LIGTYPE_OVERRIDES" \
    --all-remaining-plan-dir "$ALL_REMAINING_PLAN_DIR" \
    --all-remaining-runs-dir "$ALL_REMAINING_RUNS_DIR" \
    --all-remaining-out-dir "$ALL_REMAINING_OUT_DIR" \
    --hypothesis-version "$ALL_REMAINING_HYPOTHESIS_VERSION" \
    --remaining-scope "$ALL_REMAINING_SCOPE" \
    --long-bond-threshold "$REGEN_LONG_BOND_THRESHOLD" \
    --borderline-longish-threshold "$REGEN_BORDERLINE_LONGISH_THRESHOLD" \
    --ambiguous-gap-threshold "$REGEN_AMBIGUOUS_GAP_THRESHOLD"
  plan_rc=$?
  set -e
  if (( plan_rc != 0 )); then
    echo "All-remaining planner returned $plan_rc; no jobs were submitted." >&2
    echo "Resolve every blocked/unavailable source in $ALL_REMAINING_PLAN_DIR/manifest.csv first." >&2
    return "$plan_rc"
  fi
}

all_remaining_config_dump() {
  "$PY" - \
    "$ALL_REMAINING_PLAN_DIR/run_config.csv" \
    "$ALL_REMAINING_PLAN_DIR/manifest.csv" \
    "$ALL_REMAINING_PLAN_DIR/plan_meta.json" \
    "$ALL_REMAINING_RUNS_DIR" \
    "$ALL_REMAINING_OUT_DIR" \
    "$ALL_REMAINING_SCOPE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

config_path, manifest_path, plan_meta_path, runs_root, out_root = map(Path, sys.argv[1:6])
expected_scope = sys.argv[6]
required_routes = {
    "adaptive_cn_fill", "aminopoly_cn8", "placement_qc", "placement_build",
    "canonical_template_replan", "canonical_missing", "nitrate_parity_cn",
}
required_config = {
    "run_group", "queue", "rows", "profile_sequence", "max_attempts",
    "timeout_per_complex", "num_shards", "max_concurrent_tasks", "memory",
    "walltime", "run_id", "queue_sha256", "reports_dir", "regen_out",
}
required_queue = {
    "build_id", "root_source_build_id", "source_build_id", "rescue_route",
    "hypothesis_version",
}
required_manifest = {
    "root_source_build_id", "state", "route", "planned_build_id",
}


def resolve(path_text: object) -> Path:
    path = Path(str(path_text))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def integer(value: object, name: str, route: str) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        raise SystemExit(f"{route}: {name} is not an integer: {value!r}")
    return number


if not config_path.is_file() or not manifest_path.is_file() or not plan_meta_path.is_file():
    raise SystemExit(f"Missing immutable all-remaining plan files in {config_path.parent}")
try:
    plan_meta = json.loads(plan_meta_path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"Invalid immutable plan metadata {plan_meta_path}: {exc}")
if plan_meta.get("schema_version") != 1:
    raise SystemExit(f"Unsupported all-remaining plan metadata schema: {plan_meta.get('schema_version')!r}")
if str(plan_meta.get("remaining_scope", "")).strip() != expected_scope:
    raise SystemExit(
        "All-remaining plan scope mismatch: "
        f"plan={plan_meta.get('remaining_scope')!r} requested={expected_scope!r}"
    )
config = pd.read_csv(config_path, low_memory=False)
manifest = pd.read_csv(manifest_path, low_memory=False)
manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
if str(plan_meta.get("manifest_sha256", "")).strip() != manifest_digest:
    raise SystemExit("All-remaining plan_meta manifest_sha256 does not match manifest.csv")
if str(plan_meta.get("run_config_sha256", "")).strip() != config_digest:
    raise SystemExit("All-remaining plan_meta run_config_sha256 does not match run_config.csv")


def meta_integer(name: str) -> int:
    return integer(plan_meta.get(name), f"plan_meta.{name}", "plan")


baseline_count = meta_integer("baseline_count")
deferred_count = meta_integer("deferred_existing_successes")
manifest_count = meta_integer("manifest_count")
queued_count = meta_integer("queued_count")
coverage_total = meta_integer("coverage_total")
if manifest_count != len(manifest):
    raise SystemExit(
        f"All-remaining plan_meta manifest_count={manifest_count} but manifest has {len(manifest)} rows"
    )
if baseline_count < 0 or deferred_count < 0 or baseline_count != deferred_count + manifest_count:
    raise SystemExit(
        "All-remaining baseline coverage failed: "
        f"baseline={baseline_count} deferred={deferred_count} manifest={manifest_count}"
    )
if coverage_total != baseline_count:
    raise SystemExit(
        f"All-remaining plan_meta coverage_total={coverage_total} != baseline_count={baseline_count}"
    )
missing = sorted(required_config - set(config.columns))
if missing:
    raise SystemExit(f"All-remaining run config is missing columns: {missing}")
missing = sorted(required_manifest - set(manifest.columns))
if missing:
    raise SystemExit(f"All-remaining manifest is missing columns: {missing}")
if config["run_group"].astype(str).duplicated().any():
    raise SystemExit("All-remaining run config contains duplicate run_group values")
routes = set(config["run_group"].astype(str))
if routes != required_routes:
    raise SystemExit(
        f"All-remaining run groups differ from the supported set; "
        f"missing={sorted(required_routes-routes)} extra={sorted(routes-required_routes)}"
    )
roots = manifest["root_source_build_id"].fillna("").astype(str).str.strip()
if (roots == "").any() or roots.duplicated().any():
    raise SystemExit("All-remaining manifest has blank or duplicate root_source_build_id values")
bad_states = sorted(set(manifest["state"].astype(str)) - {"resolved_existing", "queued"})
if bad_states:
    raise SystemExit(f"All-remaining manifest contains blocked/unaccounted states: {bad_states}")
manifest_queued = int(manifest["state"].astype(str).eq("queued").sum())
configured_rows = sum(integer(value, "rows", "plan") for value in config["rows"])
if queued_count != manifest_queued or queued_count != configured_rows:
    raise SystemExit(
        "All-remaining queued coverage failed: "
        f"plan_meta={queued_count} manifest={manifest_queued} run_config={configured_rows}"
    )

runs_root = runs_root.resolve()
out_root = out_root.resolve()
all_build_ids: set[str] = set()
all_root_ids: set[str] = set()
queue_route_by_build: dict[str, str] = {}
queue_root_by_build: dict[str, str] = {}
records: list[str] = []
for row in config.to_dict("records"):
    route = str(row["run_group"])
    rows = integer(row["rows"], "rows", route)
    shards = integer(row["num_shards"], "num_shards", route)
    concurrency = integer(row["max_concurrent_tasks"], "max_concurrent_tasks", route)
    attempts = integer(row["max_attempts"], "max_attempts", route)
    timeout = integer(row["timeout_per_complex"], "timeout_per_complex", route)
    if rows < 0 or attempts < 1 or timeout < 1:
        raise SystemExit(f"{route}: invalid rows/attempts/timeout values")
    if rows == 0:
        if shards != 0 or concurrency != 0:
            raise SystemExit(f"{route}: empty queue must use zero shards and concurrency")
    elif not (1 <= shards <= rows and 1 <= concurrency <= shards):
        raise SystemExit(
            f"{route}: shards/concurrency are not clamped to rows "
            f"(rows={rows}, shards={shards}, concurrency={concurrency})"
        )
    queue_path = resolve(row["queue"])
    if not queue_path.is_file():
        raise SystemExit(f"{route}: missing queue {queue_path}")
    digest = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    expected_digest = str(row["queue_sha256"]).strip()
    if digest != expected_digest:
        raise SystemExit(
            f"{route}: immutable queue fingerprint mismatch "
            f"expected={expected_digest} actual={digest}"
        )
    queue = pd.read_csv(queue_path, low_memory=False)
    if len(queue) != rows:
        raise SystemExit(f"{route}: queue has {len(queue)} rows; config says {rows}")
    missing = sorted(required_queue - set(queue.columns))
    if missing:
        raise SystemExit(f"{route}: queue is missing audit columns: {missing}")
    if rows:
        build_ids = queue["build_id"].fillna("").astype(str).str.strip()
        root_ids = queue["root_source_build_id"].fillna("").astype(str).str.strip()
        if (build_ids == "").any() or build_ids.duplicated().any():
            raise SystemExit(f"{route}: blank or duplicate build_id values")
        if (root_ids == "").any() or root_ids.duplicated().any():
            raise SystemExit(f"{route}: blank or duplicate root_source_build_id values")
        if not queue["rescue_route"].astype(str).eq(route).all():
            raise SystemExit(f"{route}: queue contains a different rescue_route")
        overlap = (all_build_ids & set(build_ids)) | (all_root_ids & set(root_ids))
        if overlap:
            raise SystemExit(f"{route}: queue overlaps another route: {sorted(overlap)[:10]}")
        all_build_ids |= set(build_ids)
        all_root_ids |= set(root_ids)
        queue_route_by_build.update({build_id: route for build_id in build_ids})
        queue_root_by_build.update(dict(zip(build_ids, root_ids)))
    reports_dir = resolve(row["reports_dir"])
    regen_out = resolve(row["regen_out"])
    if reports_dir.parent != runs_root or reports_dir.name != route:
        raise SystemExit(f"{route}: reports_dir escapes the immutable run root: {reports_dir}")
    if regen_out.parent != out_root or regen_out.name != route:
        raise SystemExit(f"{route}: regen_out escapes the immutable output root: {regen_out}")
    run_id = str(row["run_id"]).strip()
    if not run_id or run_id.lower() == "nan":
        raise SystemExit(f"{route}: blank run_id")
    values = [
        route, str(queue_path), str(rows), str(row["profile_sequence"]),
        str(attempts), str(timeout), str(shards), str(concurrency),
        str(row["memory"]), str(row["walltime"]), run_id, digest,
        str(reports_dir), str(regen_out),
    ]
    if any("|" in value or "\n" in value for value in values):
        raise SystemExit(f"{route}: unsupported delimiter in run config")
    records.append("|".join(values))

queued = manifest[manifest["state"].astype(str).eq("queued")].copy()
manifest_builds = set(queued["planned_build_id"].fillna("").astype(str).str.strip())
if "" in manifest_builds or manifest_builds != all_build_ids:
    raise SystemExit("Manifest queued build IDs do not exactly match the immutable route queues")
for row in queued.to_dict("records"):
    build_id = str(row["planned_build_id"]).strip()
    if queue_route_by_build.get(build_id) != str(row["route"]):
        raise SystemExit(f"Manifest route mismatch for queued build_id={build_id}")
    if queue_root_by_build.get(build_id) != str(row["root_source_build_id"]).strip():
        raise SystemExit(f"Manifest root mismatch for queued build_id={build_id}")

print("\n".join(records))
PY
}

run_all_remaining_status() {
  setup_python
  "$PY" scripts/build_unique_geometries.py all-remaining-status \
    --all-remaining-plan-dir "$ALL_REMAINING_PLAN_DIR" \
    --all-remaining-runs-dir "$ALL_REMAINING_RUNS_DIR" \
    --long-bond-threshold "$REGEN_LONG_BOND_THRESHOLD" \
    --borderline-longish-threshold "$REGEN_BORDERLINE_LONGISH_THRESHOLD" \
    --ambiguous-gap-threshold "$REGEN_AMBIGUOUS_GAP_THRESHOLD"
}

submit_all_remaining_chain() {
  trap on_submit_error ERR
  require_sbatch_if_needed
  resolve_all_remaining_dirs
  prepare_all_remaining_plan

  echo "==== full immutable-plan preflight (no sbatch yet) ===="
  local config_dump
  config_dump="$(all_remaining_config_dump)"
  local -a groups=() queues=() rows_list=() profiles_list=() attempts_list=()
  local -a timeout_list=() shards_list=() concurrency_list=() memory_list=()
  local -a walltime_list=() run_id_list=() queue_hash_list=() reports_list=() out_list=()
  local line group queue rows profiles attempts timeout shards concurrency memory walltime run_id queue_hash reports_dir regen_out
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    IFS='|' read -r group queue rows profiles attempts timeout shards concurrency memory walltime run_id queue_hash reports_dir regen_out <<< "$line"
    groups+=("$group")
    queues+=("$queue")
    rows_list+=("$rows")
    profiles_list+=("$profiles")
    attempts_list+=("$attempts")
    timeout_list+=("$timeout")
    shards_list+=("$shards")
    concurrency_list+=("$concurrency")
    memory_list+=("$memory")
    walltime_list+=("$walltime")
    run_id_list+=("$run_id")
    queue_hash_list+=("$queue_hash")
    reports_list+=("$reports_dir")
    out_list+=("$regen_out")
  done <<< "$config_dump"

  local total_jobs=1 total_rows=0 input_rows i
  for ((i = 0; i < ${#groups[@]}; i++)); do
    rows="${rows_list[$i]}"
    echo
    echo "-- preflight route ${groups[$i]}: $rows rows --"
    if (( rows == 0 )); then
      continue
    fi
    INPUT_CSV="${queues[$i]}"
    EXPECTED_INPUT_ROWS="$rows"
    # Slurm --export uses commas as field delimiters. Keep the shell transport
    # colon-delimited; workers convert it back to the Python CLI's commas.
    REGEN_PROFILE_SEQUENCE="${profiles_list[$i]//,/:}"
    REGEN_MAX_ATTEMPTS="${attempts_list[$i]}"
    REGEN_TIMEOUT_PER_COMPLEX="${timeout_list[$i]}"
    REGEN_NUM_SHARDS="${shards_list[$i]}"
    REGEN_MAX_CONCURRENT_TASKS="${concurrency_list[$i]}"
    REGEN_MEM="${memory_list[$i]}"
    REGEN_WALLTIME="${walltime_list[$i]}"
    REGEN_RUN_ID="${run_id_list[$i]}"
    REGEN_QUEUE_SHA256="${queue_hash_list[$i]}"
    REGEN_REPORTS_DIR="${reports_list[$i]}"
    REGEN_OUT="${out_list[$i]}"
    validate_regen_config
    validate_regen_walltime
    input_rows="$(validate_regen_queue)"
    if [ "$input_rows" != "$rows" ]; then
      echo "${groups[$i]}: preflight row count changed: config=$rows current=$input_rows" >&2
      exit 10
    fi
    validate_regen_runtime_budget "$input_rows"
    total_rows=$((total_rows + rows))
    total_jobs=$((total_jobs + REGEN_NUM_SHARDS + 1))
  done

  echo
  echo "ALL_REMAINING_PREFLIGHT_OK plan_id=$ALL_REMAINING_PLAN_ID routes=${#groups[@]} queued_rows=$total_rows qos_job_elements=$total_jobs"
  echo "PLAN_DIR=$ALL_REMAINING_PLAN_DIR"
  echo "RUNS_DIR=$ALL_REMAINING_RUNS_DIR"
  echo "OUT_DIR=$ALL_REMAINING_OUT_DIR"
  if (( total_rows == 0 )); then
    echo "Every source already passes current dual QC; no regeneration jobs are needed."
    run_all_remaining_status
    return $?
  fi

  check_no_overlap
  check_submit_limits "$total_jobs"
  if [ "$DRY_RUN" != "1" ]; then
    ensure_repo_dirs
  fi

  FAMILY_DEPENDENCY_CHAIN=1
  local next_dependency="" previous_merge
  local -a chain_rows=()
  for ((i = 0; i < ${#groups[@]}; i++)); do
    rows="${rows_list[$i]}"
    (( rows > 0 )) || continue
    group="${groups[$i]}"
    INPUT_CSV="${queues[$i]}"
    EXPECTED_INPUT_ROWS="$rows"
    REGEN_PROFILE_SEQUENCE="${profiles_list[$i]//,/:}"
    REGEN_MAX_ATTEMPTS="${attempts_list[$i]}"
    REGEN_TIMEOUT_PER_COMPLEX="${timeout_list[$i]}"
    REGEN_NUM_SHARDS="${shards_list[$i]}"
    REGEN_MAX_CONCURRENT_TASKS="${concurrency_list[$i]}"
    REGEN_MEM="${memory_list[$i]}"
    REGEN_WALLTIME="${walltime_list[$i]}"
    REGEN_RUN_ID="${run_id_list[$i]}"
    REGEN_QUEUE_SHA256="${queue_hash_list[$i]}"
    REGEN_REPORTS_DIR="${reports_list[$i]}"
    REGEN_OUT="${out_list[$i]}"
    REGEN_START_AFTER_JOB="$next_dependency"
    REGEN_CHAIN_LABEL="all_remaining_${ALL_REMAINING_PLAN_ID}_${group}"
    previous_merge="$LAST_MERGE_REGEN_JOB_ID"
    echo
    echo "==== submit route $group (after ${next_dependency:-none}) ===="
    submit_regen_chain
    if [ "$LAST_MERGE_REGEN_JOB_ID" = "$previous_merge" ]; then
      echo "Route $group unexpectedly created no merge job." >&2
      exit 10
    fi
    chain_rows+=("$group|$LAST_REGEN_JOB_ID|$LAST_MERGE_REGEN_JOB_ID")
    next_dependency="$LAST_MERGE_REGEN_JOB_ID"
  done

  submit_sbatch "all-remaining final status" \
    --job-name=ln_all_remaining_status \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --cpus-per-task=1 \
    --mem="$ALL_REMAINING_STATUS_MEM" \
    --time="$ALL_REMAINING_STATUS_WALLTIME" \
    --chdir="$PROJECT_DIR" \
    --output=logs/ln_all_remaining_status_%j.out \
    --error=logs/ln_all_remaining_status_%j.err \
    --dependency=afterany:"$next_dependency" \
    --export=ALL,PIPELINE_STEP=all_remaining_status,ALL_REMAINING_PLAN_DIR="$ALL_REMAINING_PLAN_DIR",ALL_REMAINING_RUNS_DIR="$ALL_REMAINING_RUNS_DIR",ALL_REMAINING_OUT_DIR="$ALL_REMAINING_OUT_DIR",REGEN_LONG_BOND_THRESHOLD="$REGEN_LONG_BOND_THRESHOLD",REGEN_BORDERLINE_LONGISH_THRESHOLD="$REGEN_BORDERLINE_LONGISH_THRESHOLD",REGEN_AMBIGUOUS_GAP_THRESHOLD="$REGEN_AMBIGUOUS_GAP_THRESHOLD" \
    "$SCRIPT_PATH" worker
  local status_job="$SUBMITTED_JOB_ID"

  echo
  echo "All-remaining dependency chain"
  for line in "${chain_rows[@]}"; do
    IFS='|' read -r group regen_job merge_job <<< "$line"
    echo "  $group: array=$regen_job merge(afterany)=$merge_job"
  done
  echo "  final status(afterany): $status_job"
  echo "Status exit codes: 0=all accepted, 2=incomplete/blocked, 3=all accounted but scientific rejects remain"
  if [ "$DRY_RUN" != "1" ]; then
    print_cancel_hint
  fi
}

validate_gradual_followup() {
  "$PY" - \
    "$ADAPTIVE_REGEN_QUEUE" \
    "$ADAPTIVE_REGEN_REPORTS_DIR" \
    "$FOLLOWUP_PINCER_FAILED_INPUT" \
    "$FAMILY_PLAN_DIR" \
    "$MISSING_BUILD_QUEUE" \
    "data/processed/geometry_index_merged.csv" \
    "$FOLLOWUP_EXPECTED_ADAPTIVE_ROWS" \
    "$FOLLOWUP_EXPECTED_ADAPTIVE_EXCLUDED" \
    "$FOLLOWUP_EXPECTED_ADAPTIVE_STILL_FAILED" \
    "$FOLLOWUP_EXPECTED_FAMILY_ROWS" \
    "$FOLLOWUP_EXPECTED_FAILED_QC_DEFERRED" \
    "${FOLLOWUP_FAMILY_GROUPS[@]}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

(
    adaptive_path, adaptive_reports, pincer_path, plan_dir, missing_path,
    index_path, expected_adaptive, expected_excluded, expected_adaptive_failed,
    expected_family, expected_failed_qc, *groups,
) = sys.argv[1:]
expected_adaptive = int(expected_adaptive)
expected_excluded = int(expected_excluded)
expected_adaptive_failed = int(expected_adaptive_failed)
expected_family = int(expected_family)
expected_failed_qc = int(expected_failed_qc)


def read(path_text: str, label: str, *, allow_blank: bool = False) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        if allow_blank:
            return pd.DataFrame()
        raise SystemExit(f"Blank {label}: {path}")


def ids(frame: pd.DataFrame, column: str, label: str) -> set[str]:
    if column not in frame.columns:
        raise SystemExit(f"{label} is missing {column}")
    values = frame[column].fillna("").astype(str).str.strip()
    result = set(values[(values != "") & (values.str.lower() != "nan")])
    if len(result) != len(frame):
        raise SystemExit(f"{label} has blank or duplicate {column} values")
    return result


adaptive = read(adaptive_path, "adaptive queue")
adaptive_ids = ids(adaptive, "build_id", "adaptive queue")
adaptive_sources = ids(adaptive, "source_build_id", "adaptive queue")
if len(adaptive) != expected_adaptive:
    raise SystemExit(f"Adaptive queue has {len(adaptive)} rows; expected {expected_adaptive}")
if "rescue_route" not in adaptive or not adaptive["rescue_route"].astype(str).eq("adaptive_cn_fill").all():
    raise SystemExit("Adaptive queue contains a non-adaptive rescue route")

pincer = read(pincer_path, "pincer still-failed input")
pincer_ids = ids(pincer, "build_id", "pincer still-failed input")
if len(pincer_ids) != expected_adaptive + expected_excluded:
    raise SystemExit(
        f"Pincer still-failed input has {len(pincer_ids)} rows; "
        f"expected {expected_adaptive + expected_excluded}"
    )
if not adaptive_sources <= pincer_ids:
    raise SystemExit("Adaptive source IDs are not a subset of the pincer failures")
deferred_pincer = pincer_ids - adaptive_sources
if len(deferred_pincer) != expected_excluded:
    raise SystemExit(f"Adaptive exclusion set has {len(deferred_pincer)} rows; expected {expected_excluded}")

accepted = read(
    str(Path(adaptive_reports) / "regenerated_fail_long_bond_accepted.csv"),
    "merged adaptive accepted report",
    allow_blank=True,
)
failed = read(
    str(Path(adaptive_reports) / "regenerated_fail_long_bond_still_failed.csv"),
    "merged adaptive still-failed report",
)
accepted_ok = accepted
if not accepted.empty:
    if "qc_class" not in accepted:
        raise SystemExit("Merged adaptive accepted report is missing qc_class")
    accepted_ok = accepted[accepted["qc_class"].astype(str).eq("OK")].copy()
    if "accepted_xyz_path" not in accepted_ok:
        raise SystemExit("Merged adaptive accepted report is missing accepted_xyz_path")
    missing_accepted_xyz = [
        path for path in accepted_ok["accepted_xyz_path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing_accepted_xyz:
        raise SystemExit(f"Accepted adaptive XYZ files are absent: {missing_accepted_xyz}")
accepted_ids = set(accepted_ok.get("build_id", pd.Series(dtype=str)).astype(str))
failed_ids = set(failed.get("build_id", pd.Series(dtype=str)).astype(str)) - accepted_ids
if accepted_ids | failed_ids != adaptive_ids:
    pending = sorted(adaptive_ids - accepted_ids - failed_ids)
    unexpected = sorted((accepted_ids | failed_ids) - adaptive_ids)
    raise SystemExit(f"Adaptive merge is incomplete or stale; pending={pending} unexpected={unexpected}")
if len(failed_ids) != expected_adaptive_failed:
    raise SystemExit(
        f"Adaptive merge has {len(failed_ids)} still-failed rows; expected {expected_adaptive_failed}"
    )

config = read(str(Path(plan_dir) / "run_config.csv"), "family run config")
if not {"run_group", "rows", "num_shards"} <= set(config.columns):
    raise SystemExit("Family run config is missing required columns")
config["run_group"] = config["run_group"].astype(str)
by_group = config.set_index("run_group")
if "template_pincer" in groups or any(group not in by_group.index for group in groups):
    raise SystemExit("Follow-up family group selection is invalid")

family_build_ids: set[str] = set()
family_source_ids: set[str] = set()
family_rows = 0
family_jobs = 0
for group in groups:
    queue = read(str(Path(plan_dir) / f"{group}.csv"), f"family queue {group}")
    planned = int(float(by_group.loc[group, "rows"]))
    if len(queue) != planned:
        raise SystemExit(f"Family queue {group} has {len(queue)} rows; expected {planned}")
    build_ids = ids(queue, "build_id", f"family queue {group}")
    source_ids = ids(queue, "source_build_id", f"family queue {group}")
    if family_build_ids & build_ids or family_source_ids & source_ids:
        raise SystemExit(f"Family queue {group} overlaps an earlier group")
    family_build_ids |= build_ids
    family_source_ids |= source_ids
    family_rows += len(queue)
    if len(queue):
        family_jobs += int(float(by_group.loc[group, "num_shards"])) + 1
if family_rows != expected_family or family_jobs != 62:
    raise SystemExit(f"Family follow-up is {family_rows} rows/{family_jobs} jobs; expected {expected_family}/62")

missing = read(missing_path, "frozen missing-geometry queue")
missing_ids = ids(missing, "build_id", "frozen missing-geometry queue")
if len(missing_ids) != 12:
    raise SystemExit(f"Frozen missing-geometry queue has {len(missing_ids)} rows; expected 12")

family_all = family_build_ids | family_source_ids
adaptive_all = adaptive_ids | adaptive_sources
overlap = (
    (missing_ids & family_all)
    | (missing_ids & adaptive_all)
    | (family_all & adaptive_all)
)
if overlap:
    raise SystemExit(f"Follow-up queues overlap: {sorted(overlap)}")

index = read(index_path, "merged geometry index")
failed_qc = index[index["status"].astype(str).eq("failed_qc")]
failed_qc_ids = ids(failed_qc, "build_id", "failed_qc index rows")
if len(failed_qc_ids) != expected_failed_qc:
    raise SystemExit(f"Merged index has {len(failed_qc_ids)} failed_qc rows; expected {expected_failed_qc}")
if failed_qc_ids & (missing_ids | family_all | adaptive_all):
    raise SystemExit("Deferred failed_qc rows overlap a submitted queue")

print(
    "FOLLOWUP_PREFLIGHT_OK "
    f"adaptive={len(adaptive_ids)} adaptive_accepted={len(accepted_ids)} "
    f"adaptive_failed={len(failed_ids)} missing={len(missing_ids)} "
    f"family={family_rows} family_jobs={family_jobs} "
    f"deferred_pincer={len(deferred_pincer)} deferred_failed_qc={len(failed_qc_ids)}"
)
print("DEFERRED_PINCER_IDS=" + ",".join(sorted(deferred_pincer)))
print("DEFERRED_ADAPTIVE_FAILED_IDS=" + ",".join(sorted(failed_ids)))
print("DEFERRED_FAILED_QC_IDS=" + ",".join(sorted(failed_qc_ids)))
PY
}

submit_gradual_followup_now() {
  trap on_submit_error ERR
  require_sbatch_if_needed
  setup_python
  check_pandas_only

  local chain_id chain_dir
  chain_id="${SLURM_JOB_ID:-dryrun_$(date -u +%Y%m%dT%H%M%SZ)}"
  chain_dir="$FOLLOWUP_CHAIN_ROOT/$chain_id"
  FAMILY_PLAN_DIR="$chain_dir/family_plan"
  FAMILY_LIGTYPE_OVERRIDES="$FAMILY_PLAN_DIR/template_ligtype_overrides.csv"
  MISSING_BUILD_QUEUE="$chain_dir/missing12.csv"

  # Current hard failures: seven no-structure rows and five repeated native
  # crashes. Keep one molecule per shard and preserve each original ligand.
  MISSING_BUILD_EXPECTED_ROWS=12
  MISSING_BUILD_MAX_ROWS=12
  MISSING_BUILD_NUM_SHARDS=12
  MISSING_BUILD_MAX_CONCURRENT_TASKS=4
  MISSING_BUILD_MAX_ATTEMPTS=2
  MISSING_BUILD_TIMEOUT_PER_COMPLEX=7200
  MISSING_BUILD_MAX_TOTAL_SECONDS=39600
  MISSING_BUILD_SEED=330287
  MISSING_BUILD_PROFILE_SEQUENCE=uff_xtb_no_preopt:uff_unrelaxed
  MISSING_BUILD_WALLTIME=12:00:00
  MISSING_BUILD_MEM=5G

  echo "==== prepare immutable gradual-follow-up queues ===="
  echo "FOLLOWUP_CHAIN_DIR=$chain_dir"
  prepare_family_regen_plan
  FAMILY_PLAN_PREPARED=1
  "$PY" scripts/build_unique_geometries.py plan-missing-geometries \
    --num-shards "$MISSING_BUILD_NUM_SHARDS" \
    --expected-missing-count "$MISSING_BUILD_EXPECTED_ROWS" \
    --ligtype-overrides "$LIGTYPE_OVERRIDES" \
    --missing-queue-output "$MISSING_BUILD_QUEUE"
  validate_gradual_followup

  echo "==== combined submission preflight ===="
  echo "12 missing canonical structures + 86 non-pincer regenerations"
  echo "75 QOS job elements: 13 missing-build/QC + 62 family"
  check_no_followup_overlap
  check_no_hard10_overlap
  check_submit_limits 75

  CURRENT_STAGE="submit exact missing-geometry build"
  submit_missing_build_chain
  if [ -z "$LAST_POST_QC_JOB_ID" ]; then
    echo "Missing-geometry chain did not create a post-QC job; refusing family submission." >&2
    exit 10
  fi

  CURRENT_STAGE="submit non-pincer family dependency chain"
  REGEN_START_AFTER_JOB="$LAST_POST_QC_JOB_ID"
  submit_family_regen_dependency_chain "${FOLLOWUP_FAMILY_GROUPS[@]}"

  echo
  echo "Gradual follow-up queued"
  echo "  missing build array: $LAST_BUILD_JOB_ID"
  echo "  missing post-QC: $LAST_POST_QC_JOB_ID"
  echo "  final family merge: ${LAST_MERGE_REGEN_JOB_ID:-none}"
  echo "  deferred: $FOLLOWUP_EXPECTED_ADAPTIVE_EXCLUDED pincer-ineligible + $FOLLOWUP_EXPECTED_ADAPTIVE_STILL_FAILED adaptive-still-failed + $FOLLOWUP_EXPECTED_FAILED_QC_DEFERRED failed_qc"
  if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$chain_dir"
    {
      echo "submitted_at=$(date -Is)"
      echo "coordinator_job=${SLURM_JOB_ID:-manual}"
      echo "missing_queue=$MISSING_BUILD_QUEUE"
      echo "family_plan=$FAMILY_PLAN_DIR"
      echo "missing_build_job=$LAST_BUILD_JOB_ID"
      echo "missing_post_qc_job=$LAST_POST_QC_JOB_ID"
      echo "final_family_merge_job=${LAST_MERGE_REGEN_JOB_ID:-none}"
      echo "submitted_jobs=${CREATED_JOBS[*]}"
      echo "deferred_pincer=$FOLLOWUP_EXPECTED_ADAPTIVE_EXCLUDED"
      echo "deferred_adaptive_failed=$FOLLOWUP_EXPECTED_ADAPTIVE_STILL_FAILED"
      echo "deferred_failed_qc=$FOLLOWUP_EXPECTED_FAILED_QC_DEFERRED"
    } > "$chain_dir/submission_summary.txt"
    echo "Follow-up manifest: $chain_dir/submission_summary.txt"
  fi
}

submit_gradual_followup_coordinator() {
  local parent_job="${1:-}"
  if [ "$#" -gt 1 ] || { [ -n "$parent_job" ] && ! [[ "$parent_job" =~ ^[0-9]+$ ]]; }; then
    echo "Usage: bash slurm/geometry_pipeline.slurm submit-gradual-followup [adaptive-merge-job-id]" >&2
    exit 2
  fi

  trap on_submit_error ERR
  require_sbatch_if_needed
  ensure_repo_dirs
  check_no_followup_overlap
  check_submit_limits 1

  local -a dependency_args=()
  echo "Submitting lightweight gradual-follow-up coordinator"
  if [ -n "$parent_job" ]; then
    dependency_args+=(--dependency=afterany:"$parent_job")
    echo "Dependency: afterany:$parent_job; its preflight requires all 20 adaptive outcomes."
  else
    echo "No parent dependency: strict preflight will require all 20 adaptive outcomes before downstream submission."
  fi
  submit_sbatch "gradual follow-up coordinator" \
    --job-name=ln_geom_followup \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --cpus-per-task=1 \
    --mem="$FOLLOWUP_COORD_MEM" \
    --time="$FOLLOWUP_COORD_WALLTIME" \
    --chdir="$PROJECT_DIR" \
    --output=logs/ln_geom_followup_%j.out \
    --error=logs/ln_geom_followup_%j.err \
    ${dependency_args[@]+"${dependency_args[@]}"} \
    --export=ALL,PIPELINE_STEP=gradual_followup,DRY_RUN=0 \
    "$SCRIPT_PATH" worker
  echo "gradual follow-up coordinator job: $SUBMITTED_JOB_ID"
  echo "Watch with: squeue -u \"\$USER\""
}

submit_hard10_rescue_chain() {
  trap on_submit_error ERR
  require_sbatch_if_needed
  validate_hard10_config
  ensure_repo_dirs
  prepare_hard10_queues
  check_no_hard10_overlap

  local followup_regen_rows=0
  local planned_jobs=$((HARD10_NUM_SHARDS * 2))
  if [ "$HARD10_FOLLOWUP_REGEN" = "1" ]; then
    validate_regen_config
    validate_regen_walltime
    check_no_regen_overlap
    followup_regen_rows="$(validate_regen_queue)"
    if [ "$followup_regen_rows" = "0" ]; then
      echo "No FAIL_LONG_BOND rows in $INPUT_CSV; the hard10 chain will run without regeneration."
      HARD10_FOLLOWUP_REGEN=0
    else
      validate_regen_runtime_budget "$followup_regen_rows"
      planned_jobs=$((planned_jobs + REGEN_NUM_SHARDS + 1))
    fi
  fi
  check_submit_limits "$planned_jobs"

  local array_spec
  array_spec="0-$((HARD10_NUM_SHARDS - 1))%${HARD10_MAX_CONCURRENT_TASKS}"

  echo
  echo "Submitting isolated hard10 rescue chain"
  echo "ARRAY_SPEC=$array_spec"
  echo "HARD10_CPUS=$HARD10_CPUS"
  echo "HARD10_MEM=$HARD10_MEM"
  echo "HARD10_WALLTIME=$HARD10_WALLTIME"
  echo "HARD10_MAX_ATTEMPTS=$HARD10_MAX_ATTEMPTS"
  echo "HARD10_TIMEOUT_PER_COMPLEX=$HARD10_TIMEOUT_PER_COMPLEX"
  echo "HARD10_MAX_TOTAL_SECONDS=$HARD10_MAX_TOTAL_SECONDS"
  echo "HARD10_FOLLOWUP_REGEN=$HARD10_FOLLOWUP_REGEN"
  if [ "$HARD10_FOLLOWUP_REGEN" = "1" ]; then
    echo "FOLLOWUP_REGEN_ROWS=$followup_regen_rows"
  fi
  echo "The ordinary ln_geom job (including its 21/22 shards) is intentionally allowed to keep running."
  echo "No post-QC/canonical-index merge is submitted, so the active generation cannot race shared indexes."
  echo

  submit_sbatch "hard10 failed_no_structures array" \
    --job-name=ln_hard10_no_struct \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --array="$array_spec" \
    --cpus-per-task="$HARD10_CPUS" \
    --mem="$HARD10_MEM" \
    --time="$HARD10_WALLTIME" \
    --chdir="$PROJECT_DIR" \
    --output=logs/ln_hard10_no_struct_%A_%a.out \
    --error=logs/ln_hard10_no_struct_%A_%a.err \
    --export=ALL,PIPELINE_STEP=build,NUM_SHARDS="$HARD10_NUM_SHARDS",MAX_ATTEMPTS="$HARD10_MAX_ATTEMPTS",TIMEOUT_PER_COMPLEX="$HARD10_TIMEOUT_PER_COMPLEX",MAX_TOTAL_SECONDS="$HARD10_MAX_TOTAL_SECONDS",SEED="$HARD10_SEED",SEED_STEP="$HARD10_SEED_STEP",PROFILE_SEQUENCE="$HARD10_NO_STRUCTURES_PROFILE_SEQUENCE",SPECS_CSV="$HARD10_NO_STRUCTURES_QUEUE",FIXED_MISSING_QUEUE=1,RETRY_FAILED=0,RETRY_STATUS_LIST=,RETRY_ONLY=0,MISSING_ONLY=1,HARD_LIGAND_MODE=0,LIGTYPE_OVERRIDES="$LIGTYPE_OVERRIDES",INDEX_TAG=hard10_no_structures \
    "$SCRIPT_PATH" worker "$@"
  local no_structures_job="$SUBMITTED_JOB_ID"
  echo "failed_no_structures array job: $no_structures_job"

  submit_sbatch "hard10 native-crash array" \
    --job-name=ln_hard10_native \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --array="$array_spec" \
    --cpus-per-task="$HARD10_CPUS" \
    --mem="$HARD10_MEM" \
    --time="$HARD10_WALLTIME" \
    --chdir="$PROJECT_DIR" \
    --output=logs/ln_hard10_native_%A_%a.out \
    --error=logs/ln_hard10_native_%A_%a.err \
    --dependency=afterany:"$no_structures_job" \
    --export=ALL,PIPELINE_STEP=build,NUM_SHARDS="$HARD10_NUM_SHARDS",MAX_ATTEMPTS="$HARD10_MAX_ATTEMPTS",TIMEOUT_PER_COMPLEX="$HARD10_TIMEOUT_PER_COMPLEX",MAX_TOTAL_SECONDS="$HARD10_MAX_TOTAL_SECONDS",SEED="$HARD10_SEED",SEED_STEP="$HARD10_SEED_STEP",PROFILE_SEQUENCE="$HARD10_NATIVE_CRASH_PROFILE_SEQUENCE",SPECS_CSV="$HARD10_NATIVE_CRASH_QUEUE",FIXED_MISSING_QUEUE=1,RETRY_FAILED=0,RETRY_STATUS_LIST=,RETRY_ONLY=0,MISSING_ONLY=1,HARD_LIGAND_MODE=0,LIGTYPE_OVERRIDES="$LIGTYPE_OVERRIDES",INDEX_TAG=hard10_native_crash \
    "$SCRIPT_PATH" worker "$@"
  local native_crash_job="$SUBMITTED_JOB_ID"
  echo "native-crash array job: $native_crash_job"

  if [ "$HARD10_FOLLOWUP_REGEN" = "1" ]; then
    REGEN_START_AFTER_JOB="$native_crash_job"
    submit_regen_chain "$@"
  fi

  echo
  echo "Full dependency chain:"
  echo "  failed_no_structures  $no_structures_job  (five simultaneous tasks)"
  echo "  afterany              $native_crash_job  (five simultaneous tasks)"
  if [ "$HARD10_FOLLOWUP_REGEN" = "1" ]; then
    echo "  afterany              $LAST_REGEN_JOB_ID  (FAIL_LONG_BOND regeneration)"
    echo "  afterany              $LAST_MERGE_REGEN_JOB_ID  (partial-safe regeneration merge)"
  fi
  echo "Watch: squeue -u \$(id -un) -o '%.18i %.24j %.2t %.10M %.6D %R'"
  print_cancel_hint
}

validate_regen_queue() {
  setup_python
  if [ ! -f "$INPUT_CSV" ]; then
    echo "Missing regeneration queue: $INPUT_CSV" >&2
    exit 4
  fi

  "$PY" scripts/build_unique_geometries.py plan-regeneration-shards \
    --input "$INPUT_CSV" \
    --num-shards "$REGEN_NUM_SHARDS" >&2

  local input_rows
  input_rows="$("$PY" - "$INPUT_CSV" <<'PY'
import sys
import pandas as pd
path = sys.argv[1]
df = pd.read_csv(path, low_memory=False)
supported = pd.Series(False, index=df.index)
if "qc_class" in df.columns:
    supported |= df["qc_class"].astype(str).eq("FAIL_LONG_BOND")
if "rescue_route" in df.columns:
    routes = {
        "adaptive_cn_fill", "aminopoly_cn8", "placement_qc",
        "placement_build", "canonical_template_replan", "canonical_missing",
        "nitrate_parity_cn",
    }
    explicit = df["rescue_route"].astype(str).isin(routes)
    supported |= explicit
    if explicit.any():
        required = {"source_build_id", "hypothesis_version"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise SystemExit(f"explicit rescue input is missing audit columns: {missing}")
bad = df[~supported]
if len(bad):
    raise SystemExit(
        f"input contains {len(bad)} rows that are neither FAIL_LONG_BOND "
        "nor a supported explicit rescue hypothesis"
    )
if "build_id" in df.columns and df["build_id"].astype(str).duplicated().any():
    raise SystemExit("input contains duplicate build_id values")
print(len(df))
PY
)"
  if [ -n "$EXPECTED_INPUT_ROWS" ] && [ "$input_rows" != "$EXPECTED_INPUT_ROWS" ]; then
    echo "Input row count mismatch: got $input_rows, expected $EXPECTED_INPUT_ROWS" >&2
    exit 10
  fi

  echo "$input_rows"
}

validate_regen_runtime_budget() {
  local input_rows="$1"
  local regen_wall_seconds max_rows_per_shard worst_case_seconds required_seconds
  regen_wall_seconds="$(walltime_to_seconds "$REGEN_WALLTIME")"
  max_rows_per_shard=$(((input_rows + REGEN_NUM_SHARDS - 1) / REGEN_NUM_SHARDS))
  worst_case_seconds=$((max_rows_per_shard * REGEN_MAX_ATTEMPTS * REGEN_TIMEOUT_PER_COMPLEX))
  required_seconds=$((worst_case_seconds + REGEN_WALLTIME_MARGIN_SECONDS))
  if (( required_seconds > regen_wall_seconds )); then
    echo "Worst-case shard duration exceeds requested walltime." >&2
    echo "Rows=$input_rows max_rows_per_shard=$max_rows_per_shard worst_seconds=$worst_case_seconds margin_seconds=$REGEN_WALLTIME_MARGIN_SECONDS wall_seconds=$regen_wall_seconds" >&2
    echo "Lower REGEN_TIMEOUT_PER_COMPLEX, lower REGEN_MAX_ATTEMPTS, or increase REGEN_NUM_SHARDS within QOS limits." >&2
    exit 8
  fi
}

submit_regen_chain() {
  trap on_submit_error ERR
  require_sbatch_if_needed
  validate_regen_config
  validate_regen_walltime
  if [ "$DRY_RUN" != "1" ]; then
    ensure_repo_dirs
  fi
  if [ -n "$REGEN_START_AFTER_JOB" ]; then
    if [ "$DRY_RUN" != "1" ] && ! [[ "$REGEN_START_AFTER_JOB" =~ ^[0-9]+$ ]]; then
      echo "REGEN_START_AFTER_JOB must be a numeric SLURM job ID, got: $REGEN_START_AFTER_JOB" >&2
      exit 2
    fi
    if [ "$FAMILY_DEPENDENCY_CHAIN" != "1" ]; then
      check_no_regen_overlap
    fi
  else
    check_no_overlap
  fi
  local input_rows array_spec
  input_rows="$(validate_regen_queue)"
  if [ "$input_rows" = "0" ]; then
    echo "No supported regeneration rows in $INPUT_CSV; skipping submission."
    return 0
  fi

  if (( REGEN_NUM_SHARDS > input_rows )); then
    echo "Clamping REGEN_NUM_SHARDS from $REGEN_NUM_SHARDS to $input_rows queue rows."
    REGEN_NUM_SHARDS="$input_rows"
  fi
  if (( REGEN_MAX_CONCURRENT_TASKS > REGEN_NUM_SHARDS )); then
    echo "Clamping REGEN_MAX_CONCURRENT_TASKS from $REGEN_MAX_CONCURRENT_TASKS to $REGEN_NUM_SHARDS."
    REGEN_MAX_CONCURRENT_TASKS="$REGEN_NUM_SHARDS"
  fi
  validate_regen_config
  check_submit_limits "$((REGEN_NUM_SHARDS + 1))"

  validate_regen_runtime_budget "$input_rows"

  array_spec="0-$((REGEN_NUM_SHARDS - 1))%${REGEN_MAX_CONCURRENT_TASKS}"
  echo "Submitting focused regeneration chain"
  echo "INPUT_CSV=$INPUT_CSV"
  echo "INPUT_ROWS=$input_rows"
  echo "ARRAY_SPEC=$array_spec"
  echo "REGEN_NUM_SHARDS=$REGEN_NUM_SHARDS"
  echo "REGEN_MAX_CONCURRENT_TASKS=$REGEN_MAX_CONCURRENT_TASKS"
  echo "REGEN_WALLTIME=$REGEN_WALLTIME"
  echo "REGEN_MAX_ATTEMPTS=$REGEN_MAX_ATTEMPTS"
  echo "REGEN_TIMEOUT_PER_COMPLEX=$REGEN_TIMEOUT_PER_COMPLEX"
  echo "REGEN_WALLTIME_MARGIN_SECONDS=$REGEN_WALLTIME_MARGIN_SECONDS"
  echo "REGEN_SEED=$REGEN_SEED"
  echo "REGEN_SEED_STEP=$REGEN_SEED_STEP"
  echo "REGEN_PROFILE_SEQUENCE=$REGEN_PROFILE_SEQUENCE"
  echo "REGEN_REPORTS_DIR=$REGEN_REPORTS_DIR"
  echo "REGEN_OUT=$REGEN_OUT"
  echo "REGEN_RUN_ID=${REGEN_RUN_ID:-legacy-unversioned}"
  echo "REGEN_QUEUE_SHA256=${REGEN_QUEUE_SHA256:-computed-by-worker}"
  echo "LIGTYPE_OVERRIDES=$LIGTYPE_OVERRIDES"
  echo "REGEN_START_AFTER_JOB=${REGEN_START_AFTER_JOB:-none}"
  echo

  local -a regen_dependency_args=()
  if [ -n "$REGEN_START_AFTER_JOB" ]; then
    regen_dependency_args+=(--dependency=afterany:"$REGEN_START_AFTER_JOB")
  fi
  submit_sbatch "regeneration array" \
    --job-name=ln_regen_bad3d \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --array="$array_spec" \
    --cpus-per-task="$REGEN_CPUS" \
    --mem="$REGEN_MEM" \
    --time="$REGEN_WALLTIME" \
    --chdir="$PROJECT_DIR" \
    --output=logs/ln_regen_bad3d_%A_%a.out \
    --error=logs/ln_regen_bad3d_%A_%a.err \
    ${regen_dependency_args[@]+"${regen_dependency_args[@]}"} \
    --export=ALL,PIPELINE_STEP=regen,NUM_SHARDS="$REGEN_NUM_SHARDS",MAX_ATTEMPTS="$REGEN_MAX_ATTEMPTS",TIMEOUT_PER_COMPLEX="$REGEN_TIMEOUT_PER_COMPLEX",INPUT_CSV="$INPUT_CSV",LIGTYPE_OVERRIDES="$LIGTYPE_OVERRIDES",SEED="$REGEN_SEED",SEED_STEP="$REGEN_SEED_STEP",PROFILE_SEQUENCE="$REGEN_PROFILE_SEQUENCE",REGEN_REPORTS_DIR="$REGEN_REPORTS_DIR",REGEN_OUT="$REGEN_OUT",REGEN_RUN_ID="$REGEN_RUN_ID",REGEN_QUEUE_SHA256="$REGEN_QUEUE_SHA256",REGEN_N_SYMMETRIES="$REGEN_N_SYMMETRIES",REGEN_N_SYMMETRIES_STEP="$REGEN_N_SYMMETRIES_STEP",REGEN_N_CONFORMERS="$REGEN_N_CONFORMERS",REGEN_N_CONFORMERS_STEP="$REGEN_N_CONFORMERS_STEP",REGEN_XTB_MAX_ITERATIONS="$REGEN_XTB_MAX_ITERATIONS",REGEN_LONG_BOND_THRESHOLD="$REGEN_LONG_BOND_THRESHOLD",REGEN_BORDERLINE_LONGISH_THRESHOLD="$REGEN_BORDERLINE_LONGISH_THRESHOLD",REGEN_AMBIGUOUS_GAP_THRESHOLD="$REGEN_AMBIGUOUS_GAP_THRESHOLD" \
    "$SCRIPT_PATH" worker
  local regen_job="$SUBMITTED_JOB_ID"
  LAST_REGEN_JOB_ID="$regen_job"
  echo "regeneration array job: $regen_job"

  submit_sbatch "merge regenerated reports" \
    --job-name=ln_merge_regen \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --cpus-per-task="$MERGE_REGEN_CPUS" \
    --mem="$MERGE_REGEN_MEM" \
    --time="$MERGE_REGEN_WALLTIME" \
    --chdir="$PROJECT_DIR" \
    --output=logs/ln_merge_regen_%j.out \
    --error=logs/ln_merge_regen_%j.err \
    --dependency=afterany:"$regen_job" \
    --export=ALL,PIPELINE_STEP=merge_regen,NUM_SHARDS="$REGEN_NUM_SHARDS",MAX_ATTEMPTS="$REGEN_MAX_ATTEMPTS",TIMEOUT_PER_COMPLEX="$REGEN_TIMEOUT_PER_COMPLEX",INPUT_CSV="$INPUT_CSV",LIGTYPE_OVERRIDES="$LIGTYPE_OVERRIDES",SEED="$REGEN_SEED",SEED_STEP="$REGEN_SEED_STEP",PROFILE_SEQUENCE="$REGEN_PROFILE_SEQUENCE",REGEN_REPORTS_DIR="$REGEN_REPORTS_DIR",REGEN_OUT="$REGEN_OUT",REGEN_RUN_ID="$REGEN_RUN_ID",REGEN_QUEUE_SHA256="$REGEN_QUEUE_SHA256",REGEN_N_SYMMETRIES="$REGEN_N_SYMMETRIES",REGEN_N_SYMMETRIES_STEP="$REGEN_N_SYMMETRIES_STEP",REGEN_N_CONFORMERS="$REGEN_N_CONFORMERS",REGEN_N_CONFORMERS_STEP="$REGEN_N_CONFORMERS_STEP",REGEN_XTB_MAX_ITERATIONS="$REGEN_XTB_MAX_ITERATIONS",REGEN_LONG_BOND_THRESHOLD="$REGEN_LONG_BOND_THRESHOLD",REGEN_BORDERLINE_LONGISH_THRESHOLD="$REGEN_BORDERLINE_LONGISH_THRESHOLD",REGEN_AMBIGUOUS_GAP_THRESHOLD="$REGEN_AMBIGUOUS_GAP_THRESHOLD",FAMILY_PLAN_DIR="$FAMILY_PLAN_DIR",FAMILY_REPORTS_ROOT="$FAMILY_REPORTS_ROOT" \
    "$SCRIPT_PATH" worker
  local merge_regen_job="$SUBMITTED_JOB_ID"
  LAST_MERGE_REGEN_JOB_ID="$merge_regen_job"
  echo "merge regenerated reports job: $merge_regen_job"

  local chain_label="${REGEN_CHAIN_LABEL:-${SLURM_JOB_ID:-manual}}"
  local chain_file="logs/regen_after_qc_chain_${chain_label}.txt"
  if [ "$DRY_RUN" != "1" ]; then
    {
      echo "Delayed regeneration chain submitted at $(date)"
      echo "submitter_job=${SLURM_JOB_ID:-manual}"
      echo "regen_array_job=$regen_job"
      echo "merge_regen_job=$merge_regen_job"
      echo "num_shards=$REGEN_NUM_SHARDS"
      echo "max_concurrent_tasks=$REGEN_MAX_CONCURRENT_TASKS"
      echo "regen_walltime=$REGEN_WALLTIME"
      echo "regen_max_attempts=$REGEN_MAX_ATTEMPTS"
      echo "regen_timeout_per_complex=$REGEN_TIMEOUT_PER_COMPLEX"
      echo "regen_walltime_margin_seconds=$REGEN_WALLTIME_MARGIN_SECONDS"
      echo "cancel=scancel $regen_job $merge_regen_job"
    } > "$chain_file"
    echo "Delayed chain file: $chain_file"
  fi
}

run_build_worker() {
  ensure_repo_dirs
  setup_python
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$BUILD_CPUS}"
  export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
  export MKL_NUM_THREADS="$OMP_NUM_THREADS"

  echo "==== build geometry shard ===="
  echo "Array job: ${SLURM_ARRAY_JOB_ID:-unknown} shard: ${SLURM_ARRAY_TASK_ID:-none}/${NUM_SHARDS}"
  echo "Host: $(hostname) Date: $(date) Dir: $(pwd)"
  echo "PY=$PY"
  echo "LIGTYPE_OVERRIDES=$LIGTYPE_OVERRIDES"
  echo "RETRY_FAILED=$RETRY_FAILED"
  echo "RETRY_STATUS_LIST=$RETRY_STATUS_LIST"
  echo "RETRY_ONLY=$RETRY_ONLY"
  echo "MISSING_ONLY=$MISSING_ONLY"
  echo "MAX_TOTAL_SECONDS=$MAX_TOTAL_SECONDS"
  echo "SEED=$SEED"
  echo "SEED_STEP=$SEED_STEP"
  echo "PROFILE_SEQUENCE=$PROFILE_SEQUENCE"
  echo "SPECS_CSV=$SPECS_CSV"
  echo "FIXED_MISSING_QUEUE=$FIXED_MISSING_QUEUE"
  echo "INDEX_TAG=$INDEX_TAG"
  echo

  check_full_stack
  require_geometry_specs

  local shard_id
  shard_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  local profile_sequence
  profile_sequence="${PROFILE_SEQUENCE//:/,}"
  local geom_args=(
    --specs "$SPECS_CSV"
    --num-shards "$NUM_SHARDS"
    --shard-id "$shard_id"
    --timeout-per-complex "$TIMEOUT_PER_COMPLEX"
    --max-attempts "$MAX_ATTEMPTS"
    --max-total-seconds "$MAX_TOTAL_SECONDS"
    --seed "$SEED"
    --seed-step "$SEED_STEP"
    --profile-sequence "$profile_sequence"
    --ligtype-overrides "$LIGTYPE_OVERRIDES"
  )
  if [ "$HARD_LIGAND_MODE" = "1" ]; then
    geom_args+=(--hard-ligand-mode)
  fi
  if [ "$MISSING_ONLY" = "1" ]; then
    geom_args+=(--missing-only)
  fi
  if [ "$FIXED_MISSING_QUEUE" = "1" ]; then
    geom_args+=(--fixed-missing-queue)
  fi
  if [ -n "$INDEX_TAG" ]; then
    geom_args+=(--index-tag "$INDEX_TAG")
  fi
  if [ "$RETRY_FAILED" = "1" ]; then
    geom_args+=(--retry-failed)
  fi
  if [ -n "$RETRY_STATUS_LIST" ]; then
    IFS=',' read -r -a retry_statuses <<< "$RETRY_STATUS_LIST"
    for retry_status in "${retry_statuses[@]}"; do
      retry_status="${retry_status//[[:space:]]/}"
      if [ -n "$retry_status" ]; then
        geom_args+=(--retry-status "$retry_status")
      fi
    done
  fi
  if [ "$RETRY_ONLY" = "1" ]; then
    geom_args+=(--retry-only)
  fi

  echo "Command: $PY scripts/build_unique_geometries.py ${geom_args[*]} $*"
  "$PY" scripts/build_unique_geometries.py "${geom_args[@]}" "$@"
  echo "==== build shard $shard_id finished: $(date) ===="
}

run_post_qc_worker() {
  ensure_repo_dirs
  setup_python
  echo "==== post-build QC ===="
  echo "Job: ${SLURM_JOB_ID:-unknown}"
  echo "Host: $(hostname) Date: $(date) Dir: $(pwd)"
  echo "PY=$PY"
  echo "NUM_SHARDS=$NUM_SHARDS"
  echo "SUBMIT_REGEN_AFTER_QC=$SUBMIT_REGEN_AFTER_QC"
  echo "VERIFY_MISSING_COMPLETE=$VERIFY_MISSING_COMPLETE"
  echo

  check_pandas_only

  echo "==== recover index from disk ===="
  "$PY" scripts/build_unique_geometries.py --recover-index-only

  echo "==== merge shard indexes ===="
  "$PY" scripts/build_unique_geometries.py --merge-index-only --num-shards "$NUM_SHARDS"

  echo "==== audit xyz/index consistency ===="
  "$PY" scripts/build_unique_geometries.py --audit-xyz

  echo "==== rebuild nearest-coreCN QC queues ===="
  "$PY" scripts/build_unique_geometries.py triage-nearest-corecn

  if [ "$SUBMIT_REGEN_AFTER_QC" = "1" ]; then
    echo "==== submit delayed regeneration chain ===="
    bash "$SCRIPT_PATH" submit-regeneration
  fi

  if [ "$VERIFY_MISSING_COMPLETE" = "1" ]; then
    local remaining_missing
    remaining_missing="$("$PY" scripts/build_unique_geometries.py count-missing-geometries)"
    {
      echo "completed_at=$(date -Is)"
      echo "remaining_missing=$remaining_missing"
      echo "expected_total=$MISSING_BUILD_EXPECTED_ROWS"
    } > reports/missing_geometry_completion_summary.txt
    echo "Remaining canonical XYZ files: $remaining_missing"
    if [ "$remaining_missing" != "0" ]; then
      echo "Missing-geometry rescue incomplete: $remaining_missing canonical XYZ files are still absent." >&2
      exit 10
    fi
  fi

  echo "==== post-build QC finished: $(date) ===="
}

run_regen_worker() {
  ensure_repo_dirs
  setup_python
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$REGEN_CPUS}"
  export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
  export MKL_NUM_THREADS="$OMP_NUM_THREADS"

  echo "==== regenerate FAIL_LONG_BOND shard ===="
  echo "Array job: ${SLURM_ARRAY_JOB_ID:-unknown} shard: ${SLURM_ARRAY_TASK_ID:-none}/${NUM_SHARDS}"
  echo "Host: $(hostname) Date: $(date) Dir: $(pwd)"
  echo "PY=$PY"
  echo "INPUT_CSV=$INPUT_CSV"
  echo "MAX_ATTEMPTS=$MAX_ATTEMPTS"
  echo "TIMEOUT_PER_COMPLEX=$TIMEOUT_PER_COMPLEX"
  echo "SEED=$SEED"
  echo "SEED_STEP=$SEED_STEP"
  echo "PROFILE_SEQUENCE=$PROFILE_SEQUENCE"
  echo "REGEN_REPORTS_DIR=$REGEN_REPORTS_DIR"
  echo "REGEN_OUT=$REGEN_OUT"
  echo "REGEN_RUN_ID=${REGEN_RUN_ID:-legacy-unversioned}"
  echo "REGEN_QUEUE_SHA256=${REGEN_QUEUE_SHA256:-computed-by-worker}"
  echo "LIGTYPE_OVERRIDES=$LIGTYPE_OVERRIDES"
  echo

  check_full_stack
  require_geometry_specs
  if [ ! -f "$INPUT_CSV" ]; then
    echo "Missing regeneration queue: $INPUT_CSV" >&2
    exit 4
  fi

  "$PY" scripts/build_unique_geometries.py plan-regeneration-shards \
    --input "$INPUT_CSV" \
    --num-shards "$NUM_SHARDS"

  local shard_id
  shard_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  local profile_sequence
  profile_sequence="${PROFILE_SEQUENCE//:/,}"
  local regen_args=(
    regenerate-failed
    --input "$INPUT_CSV"
    --num-shards "$NUM_SHARDS"
    --shard-id "$shard_id"
    --max-attempts "$MAX_ATTEMPTS"
    --fixed-coordlist
    --rerun-qc
    --timeout-per-complex "$TIMEOUT_PER_COMPLEX"
    --seed "$SEED"
    --seed-step "$SEED_STEP"
    --profile-sequence "$profile_sequence"
    --n-symmetries "$REGEN_N_SYMMETRIES"
    --n-symmetries-step "$REGEN_N_SYMMETRIES_STEP"
    --n-conformers "$REGEN_N_CONFORMERS"
    --n-conformers-step "$REGEN_N_CONFORMERS_STEP"
    --xtb-max-iterations "$REGEN_XTB_MAX_ITERATIONS"
    --long-bond-threshold "$REGEN_LONG_BOND_THRESHOLD"
    --borderline-longish-threshold "$REGEN_BORDERLINE_LONGISH_THRESHOLD"
    --ambiguous-gap-threshold "$REGEN_AMBIGUOUS_GAP_THRESHOLD"
    --reports-dir "$REGEN_REPORTS_DIR"
    --regen-out "$REGEN_OUT"
    --ligtype-overrides "$LIGTYPE_OVERRIDES"
  )
  if [ -n "$REGEN_RUN_ID" ]; then
    regen_args+=(--run-id "$REGEN_RUN_ID")
  fi
  if [ -n "$REGEN_QUEUE_SHA256" ]; then
    regen_args+=(--queue-sha256 "$REGEN_QUEUE_SHA256")
  fi

  echo "Command: $PY scripts/build_unique_geometries.py ${regen_args[*]}"
  "$PY" scripts/build_unique_geometries.py "${regen_args[@]}"
  echo "==== regeneration shard $shard_id finished: $(date) ===="
}

run_merge_regen_worker() {
  ensure_repo_dirs
  setup_python
  echo "==== merge regenerated reports ===="
  echo "Job: ${SLURM_JOB_ID:-unknown}"
  echo "Host: $(hostname) Date: $(date) Dir: $(pwd)"
  echo "PY=$PY"
  echo "NUM_SHARDS=$NUM_SHARDS"
  echo "INPUT_CSV=$INPUT_CSV"
  echo "REGEN_REPORTS_DIR=$REGEN_REPORTS_DIR"
  echo "REGEN_OUT=$REGEN_OUT"
  echo "REGEN_RUN_ID=${REGEN_RUN_ID:-legacy-unversioned}"
  echo "REGEN_QUEUE_SHA256=${REGEN_QUEUE_SHA256:-computed-by-merge}"
  echo

  local profile_sequence merge_rc family_status_rc
  profile_sequence="${PROFILE_SEQUENCE//:/,}"
  local merge_args=(
    merge-regenerated
    --input "$INPUT_CSV"
    --num-shards "$NUM_SHARDS"
    --max-attempts "$MAX_ATTEMPTS"
    --timeout-per-complex "$TIMEOUT_PER_COMPLEX"
    --seed "$SEED"
    --seed-step "$SEED_STEP"
    --profile-sequence "$profile_sequence"
    --n-symmetries "$REGEN_N_SYMMETRIES"
    --n-symmetries-step "$REGEN_N_SYMMETRIES_STEP"
    --n-conformers "$REGEN_N_CONFORMERS"
    --n-conformers-step "$REGEN_N_CONFORMERS_STEP"
    --xtb-max-iterations "$REGEN_XTB_MAX_ITERATIONS"
    --long-bond-threshold "$REGEN_LONG_BOND_THRESHOLD"
    --borderline-longish-threshold "$REGEN_BORDERLINE_LONGISH_THRESHOLD"
    --ambiguous-gap-threshold "$REGEN_AMBIGUOUS_GAP_THRESHOLD"
    --reports-dir "$REGEN_REPORTS_DIR"
    --regen-out "$REGEN_OUT"
    --ligtype-overrides "$LIGTYPE_OVERRIDES"
    --allow-missing-shard-reports
  )
  if [ -n "$REGEN_RUN_ID" ]; then
    merge_args+=(--run-id "$REGEN_RUN_ID")
  fi
  if [ -n "$REGEN_QUEUE_SHA256" ]; then
    merge_args+=(--queue-sha256 "$REGEN_QUEUE_SHA256")
  fi

  set +e
  "$PY" scripts/build_unique_geometries.py "${merge_args[@]}"
  merge_rc=$?
  set -e
  family_status_rc=0
  if [[ "$REGEN_REPORTS_DIR" == "$FAMILY_REPORTS_ROOT/"* ]]; then
    set +e
    family_regen_status
    family_status_rc=$?
    set -e
  fi
  echo "==== regeneration merge finished: $(date) ===="
  if (( merge_rc != 0 )); then
    return "$merge_rc"
  fi
  return "$family_status_rc"
}

run_gradual_followup_worker() {
  ensure_repo_dirs
  echo "==== gradual follow-up coordinator ===="
  echo "Job: ${SLURM_JOB_ID:-unknown}"
  echo "Host: $(hostname) Date: $(date) Dir: $(pwd)"
  echo "Adaptive queue: $ADAPTIVE_REGEN_QUEUE"
  echo "Adaptive reports: $ADAPTIVE_REGEN_REPORTS_DIR"
  echo
  submit_gradual_followup_now
  echo "==== gradual follow-up coordinator finished: $(date) ===="
}

run_worker() {
  cd "$PROJECT_DIR"
  case "${PIPELINE_STEP:-}" in
    build)
      run_build_worker "$@"
      ;;
    post_qc)
      run_post_qc_worker "$@"
      ;;
    regen)
      run_regen_worker "$@"
      ;;
    merge_regen)
      run_merge_regen_worker "$@"
      ;;
    all_remaining_status)
      run_all_remaining_status "$@"
      ;;
    gradual_followup)
      run_gradual_followup_worker "$@"
      ;;
    *)
      echo "Unknown or missing PIPELINE_STEP=${PIPELINE_STEP:-}" >&2
      exit 2
      ;;
  esac
}

main() {
  local command="${1:-}"
  case "$command" in
    submit|full)
      shift
      submit_full_chain "$@"
      ;;
    submit-ligtype-rescue|ligtype-rescue)
      shift
      submit_ligtype_rescue_chain "$@"
      ;;
    submit-remaining|remaining)
      shift
      submit_remaining_chain "$@"
      ;;
    submit-regeneration|regen-submit)
      shift
      submit_regen_chain "$@"
      ;;
    submit-adaptive-regeneration|adaptive-regeneration)
      shift
      submit_adaptive_regen_chain "$@"
      ;;
    submit-all-remaining|all-remaining)
      shift
      submit_all_remaining_chain "$@"
      ;;
    all-remaining-status)
      shift
      if [ -z "$ALL_REMAINING_PLAN_ID" ] && [ -z "$ALL_REMAINING_PLAN_DIR" ]; then
        echo "Set ALL_REMAINING_PLAN_ID (or explicit plan/run directories) to inspect an existing immutable plan." >&2
        exit 2
      fi
      resolve_all_remaining_dirs
      run_all_remaining_status "$@"
      ;;
    submit-gradual-followup|gradual-followup)
      shift
      submit_gradual_followup_coordinator "$@"
      ;;
    dry-run-gradual-followup)
      shift
      if [ "$DRY_RUN" != "1" ]; then
        echo "dry-run-gradual-followup requires DRY_RUN=1" >&2
        exit 2
      fi
      submit_gradual_followup_now "$@"
      ;;
    submit-missing|missing|submit-missing-regeneration|missing-regeneration)
      shift
      submit_missing_build_chain "$@"
      ;;
    submit-regeneration-no-candidate|regeneration-no-candidate)
      shift
      submit_missing_regen_chain "$@"
      ;;
    prepare-family-regeneration|family-plan)
      shift
      prepare_family_regen_plan "$@"
      ;;
    family-regeneration-status|family-status)
      shift
      family_regen_status "$@"
      ;;
    submit-family-regeneration|family-regeneration)
      shift
      submit_family_regen_group "$@"
      ;;
    submit-family-regeneration-chain|family-regeneration-chain)
      shift
      submit_family_regen_dependency_chain "$@"
      ;;
    submit-hard10-rescue|hard10-rescue)
      shift
      submit_hard10_rescue_chain "$@"
      ;;
    worker)
      shift
      run_worker "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    "")
      if [ -n "${PIPELINE_STEP:-}" ] || [ -n "${SLURM_JOB_ID:-}" ]; then
        run_worker "$@"
      else
        submit_full_chain "$@"
      fi
      ;;
    *)
      if [ -n "${PIPELINE_STEP:-}" ]; then
        run_worker "$@"
      else
        usage >&2
        exit 2
      fi
      ;;
  esac
}

main "$@"
