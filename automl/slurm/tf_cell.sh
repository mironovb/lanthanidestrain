#!/bin/bash
#SBATCH --job-name=tf_cell
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_cell_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_cell_%j.err
# Cell-level set transformer (shape only): 8 seeds on the legacy population,
# 4 seeds on has3d for the held-out confirmation.  EXTRA_ARGS appended.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh; module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
python3 -u -m automl.topo.cell_transformer --population ok_only \
    --seeds ${SEEDS:-42 51 67 83 91 103 107 109} ${EXTRA_ARGS:-}
python3 -u -m automl.topo.cell_transformer --population has3d \
    --seeds ${SEEDS_H3D:-42 51 67 83} ${EXTRA_ARGS:-}
echo "TF_CELL DONE $(date -Is)"
