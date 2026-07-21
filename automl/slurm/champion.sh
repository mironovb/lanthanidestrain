#!/bin/bash
#SBATCH --job-name=ln_champ
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/champ_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/champ_%A_%a.err

# Stage E: shortlist re-run at 5 repeats x 5 folds with cluster-bootstrap CIs.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

python3 -m automl.champion \
  --out-dir "${OUT_DIR:-${REPO}/automl/artifacts/champion}" \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
  --repeats "${REPEATS:-5}" \
  --row-filter "${ROW_FILTER:-has3d}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "DONE champion ${SLURM_ARRAY_TASK_ID} $(date -Is)"
