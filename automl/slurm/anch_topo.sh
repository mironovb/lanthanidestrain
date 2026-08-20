#!/bin/bash
#SBATCH --job-name=anch_topo
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_topo_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_topo_%j.err
# Persistent-homology features (g9 persistence stats, g11 persistence images)
# given only to the shape/residual model of the anchored system.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.anchored_champion \
    --cells anch_g9 anch_g11 anch_g9_g11 --seeds 42 51 67 83
echo "ANCH_TOPO DONE $(date -Is)"
