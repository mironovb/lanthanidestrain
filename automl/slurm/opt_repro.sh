#!/bin/bash
#SBATCH --job-name=ln_repro
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/repro_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/repro_%A_%a.err
#
# Workstream A: measure the GFN2-xTB optimiser's own reproducibility from
# perturbed starts, to replace the never-measured "~0.04 A noise floor" claim
# that three published reports rest on.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="$HOME/opt/xtb-dist/bin/xtb"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_STACKSIZE=4G
cd "${REPO}"
python3 -u -m automl.qc.opt_reproducibility \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NSHARDS:-2}" \
    --workers "${WORKERS:-48}" --timeout "${TIMEOUT:-10800}" ${EXTRA:-}
echo "REPRO DONE shard=${SLURM_ARRAY_TASK_ID} rc=$? $(date -Is)"
