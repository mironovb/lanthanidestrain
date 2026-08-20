#!/bin/bash
#SBATCH --job-name=collab_rep
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/collab_rep_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/collab_rep_%j.err
# Full 5-seed reproduction of the collaborator's A2 / A2+TP / PAIRMEAN metrics.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.collab_repro --n-jobs 45
echo "COLLAB_REPRO DONE $(date -Is)"
