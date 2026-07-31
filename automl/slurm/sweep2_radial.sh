#!/bin/bash
#SBATCH --job-name=ln_sw2rad
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/sw2rad_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/sw2rad_%A_%a.err
#
# POST-HOC decomposition of sweep2 cell C1.  Not pre-registered.
#
# C1 (--radial-bins 64 --radial-max 10.0) won the screen at +0.0176, but it
# moved two things at once.  If C1 survives confirmation it becomes the study's
# one improvement to the headline metric, and an improvement whose cause is
# unidentified is a weak result.
#
#   C1BINS  64 bins, 8.0 A   -- resolution only
#   C1MAX   32 bins, 10.0 A  -- cutoff only
#
# The cutoff is the physical suspect: measured over 30,140 atoms, 24.1% lie
# beyond 8.0 A, so the published radial basis was saturating for a quarter of
# every ligand, and 10.0 A exposes an 18.4% shell previously collapsed onto the
# boundary.  Median distance-to-metal is 6.44 A and p90 is 9.34 A.
#
#   automl/slurm/campaign_driver.sh automl/slurm/sweep2_radial.sh 8 8 30
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_sweep2"
mkdir -p "${OUT}"

SEEDS=(7 11 23 37)
NS=${#SEEDS[@]}
CELLS=("C1BINS:--radial-bins 64 --radial-max 8.0"
       "C1MAX:--radial-bins 32 --radial-max 10.0")
IDX=${SLURM_ARRAY_TASK_ID}
ENTRY="${CELLS[$(( IDX / NS ))]}"
NAME="${ENTRY%%:*}"; EXTRA="${ENTRY#*:}"
S=${SEEDS[$(( IDX % NS ))]}
BASE="--arch snn --no-triangles --pair-loss-weight 2.0 --select-on adjacent --deterministic --folds 5 --repeats 3"

echo "[sw2rad] cell=${NAME} seed=${S} extra='${EXTRA}'"
python3 -u -m automl.topo.train ${BASE} --seed "${S}" ${EXTRA} \
    --tag "sw2_${NAME}_s${S}" --out-dir "${OUT}"
echo "SW2RAD DONE cell=${NAME} seed=${S} $(date -Is)"
