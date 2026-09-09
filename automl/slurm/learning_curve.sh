#!/bin/bash
#SBATCH --job-name=lcurve
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=05:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/lcurve_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/lcurve_%j.err
#
# Learning curve over training extractants for the anchored level/shape
# CatBoost (automl/topo/learning_curve.py).  ARM=both|shape_only; EXTRA for
# --smoke etc.  4 workers x 12 CatBoost threads = 48 cores.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.learning_curve --arm "${ARM:-both}" --workers "${WORKERS:-4}" --seed "${SEED:-42}" ${EXTRA:-}
echo "LCURVE DONE arm=${ARM:-both} rc=$? $(date -Is)"
