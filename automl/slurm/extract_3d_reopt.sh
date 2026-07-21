#!/bin/bash
#SBATCH --job-name=ln_x3dre
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/x3dre_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/x3dre_%A_%a.err
# Stage 2: re-extract the same 293 descriptors from the re-optimised geometries.
#
# geom3d_features.py is reused UNCHANGED apart from an added --geom-root, so any
# difference in the resulting descriptors is attributable to the geometry and
# not to the featuriser.  --geom-root is what makes that true: the re-optimised
# files deliberately reuse the original basenames, and the resolver is
# first-match-wins over an unordered rglob, so without an explicit root a run
# could silently featurise a mixture of loose and tight structures.
#
# SOLVENT selects which set to featurise; pass it in the environment:
#   SOLVENT=water   sbatch automl/slurm/extract_3d_reopt.sh
#   SOLVENT=octanol sbatch automl/slurm/extract_3d_reopt.sh
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
SOLVENT="${SOLVENT:-water}"
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

GEOMROOT="${REPO}/automl/artifacts/geom_reopt/${SOLVENT}"
OUTDIR="${REPO}/automl/artifacts/geom3d_reopt/${SOLVENT}"
mkdir -p "${OUTDIR}"

echo "[x3dre] solvent=${SOLVENT} shard=${SLURM_ARRAY_TASK_ID} root=${GEOMROOT}"
python3 -m automl.geom3d_features \
    --geom-root "${GEOMROOT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --out "${OUTDIR}/geom3d_shard${SLURM_ARRAY_TASK_ID}.parquet"

echo "X3DRE DONE solvent=${SOLVENT} shard=${SLURM_ARRAY_TASK_ID} $(date -Is)"
