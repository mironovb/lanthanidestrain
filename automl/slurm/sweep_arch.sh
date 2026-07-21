#!/bin/bash
#SBATCH --job-name=ln_arch
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/arch_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/arch_%A_%a.err

# Stage B1: feature-block ablation under leave-extractants-out CV.
set -euo pipefail

REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONWARNINGS=ignore

OUT="${OUT_DIR:-${REPO}/automl/artifacts/sweeps/arch}"
mkdir -p "${OUT}"
cd "${REPO}"

python3 -m automl.run_sweep \
    --mode arch \
    --out-dir "${OUT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --n-jobs "${SLURM_CPUS_PER_TASK}" \
    --n-splits "${N_SPLITS:-5}" \
    --repeats "${REPEATS:-3}" \
    --models "${MODELS:-lgbm}" \
    --row-filters "${ROW_FILTERS:-has3d}" \
    --weight-scheme "${WEIGHT_SCHEME:-none}" \
    --presets "${PRESETS:-baseline_2d,inner_sphere,selectivity,all_new_3d,plus_g2,plus_g10}" \
    --save-oof

echo "DONE arch shard ${SLURM_ARRAY_TASK_ID} $(date -Is)"
