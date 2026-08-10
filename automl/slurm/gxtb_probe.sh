#!/bin/bash
#SBATCH --job-name=gxtb_probe
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtb_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtb_%A_%a.err
#
# Does a better electronic-structure method see f-shell structure GFN2 cannot?
#
# GFN2 interpolates every lanthanide parameter linearly between two fitted
# anchors, so its metal response is rank-1 in Z BY CONSTRUCTION.  g-xTB puts the
# f electrons in the valence.  This probes both at a FIXED geometry, so every
# difference between the arms is electronic structure and nothing else.
#
# NOTE the binary differs from every other job in this repo: only the
# 6.7.1-gxtb build understands --gxtb.  It runs GFN2 too, so both arms come from
# one executable and the comparison carries no build confound.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS="${THREADS:-16}" MKL_NUM_THREADS="${THREADS:-16}"
export OMP_STACKSIZE=4G
cd "${REPO}"
python3 -u -m automl.qc.gxtb_probe \
    --anchor "${ANCHOR}" \
    --method "${METHOD:-both}" \
    --threads "${THREADS:-16}" \
    --tag "${TAG:-probe}" \
    ${SOLVENT:+--solvent "${SOLVENT}"} \
    ${EXTRA:-}
echo "GXTB DONE tag=${TAG:-probe} rc=$? $(date -Is)"
