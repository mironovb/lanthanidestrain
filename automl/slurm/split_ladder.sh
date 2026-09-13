#!/bin/bash
#SBATCH --job-name=split_ladder
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/split_ladder_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/split_ladder_%j.err
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh; module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.split_ladder
echo "LADDER DONE $(date -Is)"
