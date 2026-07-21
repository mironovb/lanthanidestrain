#!/bin/bash
#SBATCH --job-name=ln_optuna
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=16:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/optuna_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/optuna_%A_%a.err

# Stage C: distributed Optuna HPO.  Every array task joins the same study via a
# shared SQLite storage on the parallel filesystem, so trials are pooled.
set -euo pipefail

REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

# Each array task takes one (preset, model, objective) combination from a list,
# cycling so several workers share each study.
IFS=';' read -ra COMBOS <<< "${COMBOS:-all_3d,lgbm,r2_overall;all_3d,catboost,r2_overall;all_3d,xgb,r2_overall;baseline_2d,lgbm,r2_overall;all_3d,lgbm,r2_within;all_3d,catboost,r2_within;selectivity,lgbm,r2_within;all_3d,mlp,r2_overall}"
N=${#COMBOS[@]}
IDX=$(( SLURM_ARRAY_TASK_ID % N ))
IFS=',' read -r PRESET MODEL OBJ <<< "${COMBOS[$IDX]}"

OUT="${OUT_DIR:-${REPO}/automl/artifacts/sweeps/optuna}"
mkdir -p "${OUT}"
cd "${REPO}"

echo "worker ${SLURM_ARRAY_TASK_ID} -> preset=${PRESET} model=${MODEL} objective=${OBJ}"

python3 -m automl.run_sweep \
    --mode optuna \
    --out-dir "${OUT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --preset "${PRESET}" \
    --model "${MODEL}" \
    --objective "${OBJ}" \
    --n-trials "${N_TRIALS:-60}" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --n-splits "${N_SPLITS:-5}" \
    --repeats "${REPEATS:-2}" \
    --weight-schemes "${WEIGHT_SCHEMES:-none,group_inv,target_lds,combo}" \
    --row-filters "${ROW_FILTERS:-has3d}"

echo "DONE optuna shard ${SLURM_ARRAY_TASK_ID} $(date -Is)"
