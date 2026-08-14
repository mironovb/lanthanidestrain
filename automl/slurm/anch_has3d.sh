#!/bin/bash
#SBATCH --job-name=anch_h3d
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_h3d_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_h3d_%j.err
# Expanded-population anchored champion: OOF covers the fresh-444 rows.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.anchored_champion --population has3d \
    --cells anch_q60_mae_w07 flat_q60 --seeds 42 51 67 83
echo "ANCH_H3D DONE $(date -Is)"
