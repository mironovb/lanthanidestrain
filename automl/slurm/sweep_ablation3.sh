#!/bin/bash
#SBATCH --job-name=ln_ablate
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/ablate3_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/ablate3_%A_%a.err

# Stage B1: feature-block ablation under leave-extractants-out CV.
set -euo pipefail

REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

OUT="${OUT_DIR:-${REPO}/automl/artifacts/sweeps/ablation3}"
mkdir -p "${OUT}"
cd "${REPO}"

python3 -m automl.run_sweep \
    --mode ablation \
    --out-dir "${OUT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --n-splits "${N_SPLITS:-5}" \
    --repeats "${REPEATS:-3}" \
    --models "${MODELS:-lgbm}" \
    --row-filters "${ROW_FILTERS:-has3d,ok_only}" \
    --weight-scheme "${WEIGHT_SCHEME:-none}" \
    --presets "${PRESETS:-baseline_2d,core3d,core3d_qc,core3d_g5,core3d_ligand,core3d_smooth,core3d_all,plus_g_core,denoised,best_guess}" \
    --save-oof

echo "DONE ablation shard ${SLURM_ARRAY_TASK_ID} $(date -Is)"
