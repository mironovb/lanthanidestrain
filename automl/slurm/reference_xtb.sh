#!/bin/bash
#SBATCH --job-name=ln_refxtb
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/refxtb_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/refxtb_%A_%a.err
#
# Reference xTB energetics -- the queued-but-never-computed feature block.
#
#   MODE=probe sbatch automl/slurm/reference_xtb.sh                 # RUN THIS FIRST
#   MODE=run   sbatch --array=0-15 automl/slurm/reference_xtb.sh
#   MODE=collect sbatch automl/slurm/reference_xtb.sh
#
# The probe is not optional and not a formality.  It substitutes all 14
# lanthanides into the same frozen cage and measures how far GFN2 moves the
# energy.  If the method cannot resolve adjacent lanthanides, no descriptor
# built on it can carry the adjacent-pair signal, and that is worth twenty
# minutes to establish rather than eighty CPU-hours to discover.
#
# OMP_NUM_THREADS=1: parallelism is across structures, not inside xtb, exactly
# as reopt.sh does it -- xtb scales poorly past a few threads and there are
# hundreds of independent structures.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="${HOME}/opt/xtb-dist/bin/xtb"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${REPO}"

MODE="${MODE:?set MODE to probe, run or collect}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

case "${MODE}" in
  probe)
    python3 -u -m automl.qc.reference_xtb --probe --probe-n "${PROBE_N:-12}" \
        --workers "${SLURM_CPUS_PER_TASK}" --timeout 1800
    ;;
  run)
    python3 -u -m automl.qc.reference_xtb --run \
        --shard "${SHARD}" --n-shards "${NSHARDS}" \
        --workers "${SLURM_CPUS_PER_TASK}" --timeout 3600
    ;;
  collect)
    python3 -u -m automl.qc.reference_xtb --collect
    ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 1 ;;
esac
RC=$?
echo "REFXTB DONE mode=${MODE} shard=${SHARD} rc=${RC} $(date -Is)"
exit "${RC}"
