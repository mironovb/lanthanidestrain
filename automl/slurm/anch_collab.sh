#!/bin/bash
#SBATCH --job-name=anch_col
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_col_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_col_%j.err
# Our anchored champion trained on the collaborator's expanded population.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.anchored_champion --population collab \
    --cells anch_q60_q60 flat_q60 --seeds 42 51 67 83
echo "ANCH_COLLAB DONE $(date -Is)"
