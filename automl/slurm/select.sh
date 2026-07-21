#!/bin/bash
#SBATCH --job-name=ln_select
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/select_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/select_%A_%a.err

# Stage B3: automated feature selection.
#   tasks 0-3: greedy forward block search for four different objectives
#   tasks 4-7: permutation importance for the reference block sets
set -euo pipefail

REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

OUT="${OUT_DIR:-${REPO}/automl/artifacts/selection}"
mkdir -p "${OUT}"
PARAMS='{"n_estimators":700,"learning_rate":0.05,"num_leaves":63,"colsample_bytree":0.5,"reg_lambda":5.0}'

case "${SLURM_ARRAY_TASK_ID}" in
  0) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_overall --model lgbm --params "${PARAMS}" \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 2 ;;
  1) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_within --model lgbm --params "${PARAMS}" \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 2 ;;
  2) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective sel_logSF_r2 --model lgbm --params "${PARAMS}" \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 2 ;;
  3) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_within_composition --model lgbm --params "${PARAMS}" \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 2 ;;
  4) python3 -m automl.select_features --task importance --out-dir "${OUT}" \
        --blocks "rdkit,ecfp,metal,cond,plan,qc,g1,g2,g3,g4,g5,g7,g8,g9" \
        --model lgbm --params "${PARAMS}" --row-filter has3d \
        --n-jobs "${SLURM_CPUS_PER_TASK}" ;;
  5) python3 -m automl.select_features --task importance --out-dir "${OUT}" \
        --blocks "metal,cond,qc,g1,g2,g3,g4,g5,g6,g7,g8,g9" \
        --model lgbm --params "${PARAMS}" --row-filter has3d \
        --n-jobs "${SLURM_CPUS_PER_TASK}" ;;
  6) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_overall --model lgbm --params "${PARAMS}" \
        --row-filter ok_only --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 2 ;;
  7) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_overall --model catboost --params '{}' \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 2 ;;
esac

echo "DONE select ${SLURM_ARRAY_TASK_ID} $(date -Is)"
