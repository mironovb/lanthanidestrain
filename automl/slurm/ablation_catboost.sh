#!/bin/bash
#SBATCH --job-name=ln_abcat
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/abcat_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/abcat_%A_%a.err

# The LightGBM ablation ranked the 3D blocks, but the model sweep then showed
# CatBoost + inverse-extractant weighting is a much stronger base learner
# (R2 0.518 vs 0.459 on the same 2D features).  Block conclusions must be
# re-checked on the stronger learner before they are reported as final.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

python3 -m automl.run_sweep \
  --mode ablation \
  --out-dir "${OUT_DIR:-${REPO}/automl/artifacts/sweeps/ablation_catboost}" \
  --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" --n-splits 5 --repeats "${REPEATS:-3}" \
  --models catboost \
  --weight-scheme "${WEIGHT_SCHEME:-group_inv}" \
  --row-filters "${ROW_FILTERS:-has3d}" \
  --presets "${PRESETS:-baseline_2d,plus_g5,plus_g1,plus_g14c,plus_g13c,plus_g15c,ligand3d_only,denoised,cnfree_ligand,g5_ligand,core3d_qc,inner_sphere,plus_g10,all_3d}" \
  --save-oof
echo "DONE abcat shard ${SLURM_ARRAY_TASK_ID} $(date -Is)"
