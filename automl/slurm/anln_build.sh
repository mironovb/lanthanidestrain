#!/bin/bash
#SBATCH --job-name=anln_build
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anln_build_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/anln_build_%j.err
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.an_ln.build
echo "ANLN_BUILD DONE $(date -Is)"
