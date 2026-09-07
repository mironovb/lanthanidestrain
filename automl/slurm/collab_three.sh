#!/bin/bash
#SBATCH --job-name=collab_three
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/collab_three_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/collab_three_%j.err
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.collab_three_models --seeds 16
echo "COLLAB_THREE DONE $(date -Is)"
