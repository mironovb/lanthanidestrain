#!/bin/bash
#SBATCH --job-name=anch_night
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_night_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_night_%j.err
# Night tests: (a) expanded-population anchored winner to 8 seeds,
# (b) shape-weight refinement of the winner on the legacy population.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.anchored_champion --population has3d \
    --cells anch_q60_q60 --seeds 91 103 107 109
python3 -u -m automl.topo.anchored_champion \
    --cells anch_q60_q60_w08 anch_q60_q60_w09 --seeds 42 51 67 83
echo "ANCH_NIGHT DONE $(date -Is)"
