#!/bin/bash
#SBATCH --job-name=ln_models
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/models_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/models_%A_%a.err

# Stage B2: model family x sample-weight scheme sweep at fixed feature preset.
set -euo pipefail

REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

OUT="${OUT_DIR:-${REPO}/automl/artifacts/sweeps/models}"
mkdir -p "${OUT}"
cd "${REPO}"

python3 -m automl.run_sweep \
    --mode models \
    --out-dir "${OUT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --n-splits "${N_SPLITS:-5}" \
    --repeats "${REPEATS:-2}" \
    --presets "${PRESETS:-baseline_2d,all_3d}" \
    --models "${MODELS:-lgbm,xgb,catboost,hgb,rf,extratrees,ridge,elasticnet,svr,knn,mlp,huber}" \
    --weight-schemes "${WEIGHT_SCHEMES:-none,group_inv,target_lds,combo}" \
    --row-filters "${ROW_FILTERS:-has3d}"

echo "DONE models shard ${SLURM_ARRAY_TASK_ID} $(date -Is)"
