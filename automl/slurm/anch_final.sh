#!/bin/bash
#SBATCH --job-name=anch_final
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_final_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_final_%j.err
# Final robustness: 4 more seeds on the winning cell (8-seed ensemble total),
# and the winning cell on the expanded population for fresh-444 confirmation.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.anchored_champion --cells anch_q60_q60 --seeds 91 103 107 109
python3 -u -m automl.topo.anchored_champion --population has3d --cells anch_q60_q60 --seeds 42 51 67 83
echo "ANCH_FINAL DONE $(date -Is)"
