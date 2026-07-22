#!/bin/bash
#SBATCH --job-name=ln_pigeom
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pigeom_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pigeom_%j.err
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -m automl.qc.pi_sweep_geometry --stage a
echo "PIGEOM DONE $(date -Is)"
