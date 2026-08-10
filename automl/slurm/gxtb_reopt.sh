#!/bin/bash
#SBATCH --job-name=gxtb_ro
#SBATCH --nodes=1
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtbro_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtbro_%A_%a.err
#
# Re-optimise the REAL 956 dataset complexes with g-xTB so the modelling
# question gets a score instead of a proxy.  Gas phase: g-xTB's ddCOSMO arm
# failed 14-23% of optimisations with the failures concentrated on particular
# metals, and metal-correlated missingness is fatal to a selectivity study.
# Records are one atomic JSON per complex, so a job killed at the wall clock
# keeps everything it finished and a rerun is idempotent.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_STACKSIZE=4G
cd "${REPO}"
python3 -u -m automl.qc.gxtb_reopt \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NSHARDS:-6}" \
    --workers "${WORKERS:-48}" --timeout "${TIMEOUT:-14400}" ${EXTRA:-}
echo "GXTBRO DONE shard=${SLURM_ARRAY_TASK_ID} rc=$? $(date -Is)"
