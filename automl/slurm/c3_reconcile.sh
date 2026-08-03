#!/bin/bash
#SBATCH --job-name=ln_c3rec
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c3rec_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c3rec_%A_%a.err
#
# POST-HOC for campaign 3.  Not pre-registered; designed after the screen.
#
# The screen was monotone in the wrong direction -- T2 -0.0253, T2W -0.0321,
# T2X -0.0832 -- and the diagnosis is that the pair head's skill never reaches
# the metric: the metric differences LEVEL-head predictions, and the pair head
# is on a pathway evaluation never touches.
#
# --pair-reconcile routes it in at inference.  Two configs, because which pair
# head is better trained is itself unknown:
#   T2REC   pair head alongside the surrogate (w=1)
#   T2XREC  pair head alone (w=2), the best-trained pair head but the worst
#           level head
#
# If reconciliation helps, the pair head had skill the metric was discarding.
# If it hurts, the pair head has no dy skill worth routing and T2's failure is
# not a plumbing problem.  Either way the mechanism claim is settled.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_c3"; mkdir -p "${OUT}"
SEEDS=(7 11 23 37); NS=${#SEEDS[@]}
BASE="--arch snn --no-triangles --select-on adjacent --deterministic --folds 5 --repeats 3"
CELLS=(
  "T2REC:--pair-loss-weight 2.0 --pair-head --pair-head-weight 1.0 --pair-reconcile"
  "T2XREC:--pair-loss-weight 0.0 --pair-head --pair-head-weight 2.0 --pair-reconcile"
)
IDX=${SLURM_ARRAY_TASK_ID}
ENTRY="${CELLS[$(( IDX / NS ))]}"
NAME="${ENTRY%%:*}"; EXTRA="${ENTRY#*:}"
S=${SEEDS[$(( IDX % NS ))]}
echo "[c3rec] cell=${NAME} seed=${S}"
python3 -u -m automl.topo.train ${BASE} ${EXTRA} --seed "${S}" \
    --tag "c3_${NAME}_s${S}" --out-dir "${OUT}"
echo "C3REC DONE cell=${NAME} seed=${S} $(date -Is)"
