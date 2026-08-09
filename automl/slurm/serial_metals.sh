#!/bin/bash
#SBATCH --job-name=ln_serial
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/serial_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/serial_%A_%a.err
#
# Workstream B: build the lanthanide series IN CORRESPONDENCE by metal
# substitution from one relaxed anchor per family.  Sharded BY FAMILY so a
# family is never split across jobs -- its members share one anchor read.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="$HOME/opt/xtb-dist/bin/xtb"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_STACKSIZE=4G
cd "${REPO}"
python3 -u -m automl.qc.serial_metals --mode "${MODE:-serial}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NSHARDS:-2}" \
    --workers "${WORKERS:-48}" --timeout "${TIMEOUT:-10800}" ${EXTRA:-}
echo "SERIAL DONE mode=${MODE:-serial} shard=${SLURM_ARRAY_TASK_ID} rc=$? $(date -Is)"
