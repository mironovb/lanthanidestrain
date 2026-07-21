#!/bin/bash
#SBATCH --job-name=ln_geom3d
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-31
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/geom3d_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/geom3d_%A_%a.err

# Stage A: extract rich 3D descriptors from every GFN2-xTB optimised complex
# geometry that exists on disk.  Read-only with respect to data/; all output
# goes to automl/artifacts/geom3d/.
set -euo pipefail

REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

OUTDIR="${REPO}/automl/artifacts/geom3d"
mkdir -p "${OUTDIR}"

cd "${REPO}"
python3 -m automl.geom3d_features \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --out "${OUTDIR}/geom3d_shard${SLURM_ARRAY_TASK_ID}.parquet"

echo "DONE shard ${SLURM_ARRAY_TASK_ID} $(date -Is)"
