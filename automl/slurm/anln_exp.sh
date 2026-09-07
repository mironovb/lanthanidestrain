#!/bin/bash
#SBATCH --job-name=anln_exp
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=10:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anln_exp_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anln_exp_%j.err
# An/Ln transfer: lanthanide-only training (zero-shot Am/Eu) then joint training.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.an_ln.experiments --config ln_only
python3 -u -m automl.an_ln.experiments --config ln_am
echo "ANLN_EXP DONE $(date -Is)"
