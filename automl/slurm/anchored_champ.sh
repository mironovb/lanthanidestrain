#!/bin/bash
#SBATCH --job-name=anch_champ
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_champ_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anch_champ_%j.err
# A1 follow-up: anchored architecture x champion quantile loss (never combined).
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.anchored_champion
echo "ANCH_CHAMP DONE $(date -Is)"
