#!/bin/bash
#SBATCH --job-name=ln_select2
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/select2_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/select2_%A_%a.err

# Stage B4: second selection round, now that the denoised (g12c/g13c/g14c) and
# curated (g_core) blocks exist and the first importance pass has run.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

OUT="${OUT_DIR:-${REPO}/automl/artifacts/selection2}"
mkdir -p "${OUT}"
PARAMS='{"n_estimators":700,"learning_rate":0.05,"num_leaves":63,"colsample_bytree":0.5,"reg_lambda":5.0}'

case "${SLURM_ARRAY_TASK_ID}" in
  0) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_overall --model lgbm --params "${PARAMS}" \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 3 ;;
  1) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_within --model lgbm --params "${PARAMS}" \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 3 ;;
  2) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective sel_logSF_r2 --model lgbm --params "${PARAMS}" \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 3 ;;
  3) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_within_composition --model lgbm --params "${PARAMS}" \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 3 ;;
  4) python3 -m automl.select_features --task importance --out-dir "${OUT}" \
        --blocks "rdkit,ecfp,metal,cond,plan,qc,g_core,g12c,g13c,g14c" \
        --model lgbm --params "${PARAMS}" --row-filter has3d \
        --n-jobs "${SLURM_CPUS_PER_TASK}" ;;
  5) python3 -m automl.select_features --task greedy --out-dir "${OUT}" \
        --objective r2_overall --model catboost --params '{}' \
        --row-filter has3d --n-jobs "${SLURM_CPUS_PER_TASK}" --repeats 2 ;;
esac
echo "DONE select2 ${SLURM_ARRAY_TASK_ID} $(date -Is)"
