#!/bin/bash
#SBATCH --job-name=ln_c3
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c3_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c3_%A_%a.err
#
# Pre-registered in automl/reports/CAMPAIGN3_PREREGISTRATION.md.
# T1 closed on CPU (CEILING_CLOSED.md); T4 reported NOT TESTED (speciation.py).
#
#   automl/slurm/campaign_driver.sh automl/slurm/campaign3.sh 24 8 30
#   automl/slurm/campaign_driver.sh automl/slurm/campaign3.sh 24 8 30 MODE=confirm CELL=T2
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_c3"
mkdir -p "${OUT}"

SEEDS=(7 11 23 37)
CONFIRM_SEEDS=(101 103 107 109 113 127 131 137 139 149 151 157)
NS=${#SEEDS[@]}
BASE="--arch snn --no-triangles --select-on adjacent --deterministic --folds 5 --repeats 3"

# D0 is the sweep2 anchor, unchanged, so campaign 3 is directly comparable.
# T2*  give the pair difference its own parameters.
# T2X  removes the scalar surrogate, isolating the parametrised difference.
# T3   lets 45 diluents and 9 acids reach the structural embedding.
CELLS=(
  "D0:--pair-loss-weight 2.0"
  "T2:--pair-loss-weight 2.0 --pair-head --pair-head-weight 1.0"
  "T2W:--pair-loss-weight 2.0 --pair-head --pair-head-weight 3.0"
  "T2X:--pair-loss-weight 0.0 --pair-head --pair-head-weight 2.0"
  "T3:--pair-loss-weight 2.0 --film"
  "T23:--pair-loss-weight 2.0 --pair-head --pair-head-weight 1.0 --film"
)

if [[ "${MODE:-run}" == "confirm" ]]; then
  : "${CELL:?MODE=confirm needs CELL=<winning cell>}"
  NCS=${#CONFIRM_SEEDS[@]}
  IDX=${SLURM_ARRAY_TASK_ID}
  if (( IDX < NCS )); then NAME="D0"; else NAME="${CELL}"; fi
  S=${CONFIRM_SEEDS[$(( IDX % NCS ))]}
  EXTRA=""
  for E in "${CELLS[@]}"; do [[ "${E%%:*}" == "${NAME}" ]] && EXTRA="${E#*:}"; done
  echo "[c3-confirm] cell=${NAME} seed=${S}"
  python3 -u -m automl.topo.train ${BASE} ${EXTRA} --seed "${S}" \
      --tag "c3_${NAME}_s${S}" --out-dir "${OUT}"
  echo "C3 DONE cell=${NAME} seed=${S} $(date -Is)"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID}
ENTRY="${CELLS[$(( IDX / NS ))]}"
NAME="${ENTRY%%:*}"; EXTRA="${ENTRY#*:}"
S=${SEEDS[$(( IDX % NS ))]}
echo "[c3] cell=${NAME} seed=${S} extra='${EXTRA}'"
python3 -u -m automl.topo.train ${BASE} ${EXTRA} --seed "${S}" \
    --tag "c3_${NAME}_s${S}" --out-dir "${OUT}"
echo "C3 DONE cell=${NAME} seed=${S} $(date -Is)"
