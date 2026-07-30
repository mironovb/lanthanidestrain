#!/bin/bash
#SBATCH --job-name=ln_sweep2
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/sw2_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/sw2_%A_%a.err
#
# Pre-registered in automl/reports/SWEEP2_PREREGISTRATION.md, committed before
# the first run existed.
#
#   MODE=timing sbatch automl/slurm/sweep2.sh            # RUN THIS FIRST
#   automl/slurm/campaign_driver.sh automl/slurm/sweep2.sh 44 8 34
#
#   # confirmatory stage -- ONLY if a cell clears the pre-registered +0.005
#   # tune-half gate.  CELL is the screen winner.
#   automl/slurm/campaign_driver.sh automl/slurm/sweep2.sh 24 8 34 \
#       MODE=confirm CELL=A2
#
# 11 cells x 4 seeds.  Base is G0 (--no-triangles): better than the published
# simplicial arm, faster, and much cheaper to run deterministically because the
# float64 sorted scatter is dominated by the 9.3M triangles it does not have.
#
# --deterministic is ON.  This is the first sweep in the study that can select:
# the persistence-image sweep could not rank 25 configurations because re-running
# one cell moved it by more than the differences being compared.  Bit-identical
# runs mean 4 seeds give an exact ranking.
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

# cell -> extra flags.  A0 is the anchor and carries none.
CELLS=(
  "A0:"
  "A1:--preset baseline_2d_shape"
  "A2:--node-angular"
  "A3:--angular-readout"
  "B1:--aux-target cshm"
  "B2:--aux-target eint"
  "B3:--aux-target qtransfer"
  "C1:--radial-bins 64 --radial-max 10.0"
  "C2:--attn-pool"
  "C3:--lr 5e-4"
  "C4:--weight-decay 1e-3"
)

BASE="--arch snn --no-triangles --pair-loss-weight 2.0 --select-on adjacent --deterministic --folds 5 --repeats 3"

if [[ "${MODE:-run}" == "timing" ]]; then
  # Measure the deterministic slowdown on G0 before committing 44 runs to a
  # guessed 4x.  One fold, one seed.
  echo "=== timing: G0 non-deterministic vs deterministic, 5 folds x 1 repeat ==="
  for FLAG in "" "--deterministic"; do
    S=$SECONDS
    python3 -u -m automl.topo.train --arch snn --no-triangles \
        --pair-loss-weight 2.0 --select-on adjacent --folds 5 --repeats 1 \
        --seed 7 ${FLAG} --tag "timing$(echo ${FLAG} | tr -d ' -')" \
        --out-dir "${OUT}/_timing" >/dev/null 2>&1 || true
    echo "  flag='${FLAG:-none}'  elapsed $((SECONDS-S)) s"
  done
  echo "TIMING DONE $(date -Is)"
  exit 0
fi

# Confirmatory stage: the winner and the A0 anchor, both at 16 seeds, scored
# once on the 78 held-out confirm extractants under both block keys.
#
# 24 tasks, not 32, because the four screening seeds are REUSED.  That is
# legitimate and worth stating: selection was made on the 84 tune extractants
# only, and the confirm half was never looked at, so the confirm-half score of
# an already-existing run is still a first look at those rows.  Re-running them
# would cost 8 GPU runs to obtain bit-identical predictions -- the runs are
# deterministic, so the re-run would return exactly the same numbers.
CONFIRM_SEEDS=(101 103 107 109 113 127 131 137 139 149 151 157)
if [[ "${MODE:-run}" == "confirm" ]]; then
  : "${CELL:?MODE=confirm needs CELL=<winning cell>}"
  NCS=${#CONFIRM_SEEDS[@]}
  IDX=${SLURM_ARRAY_TASK_ID}
  # first NCS tasks extend A0, the rest extend the winner
  if (( IDX < NCS )); then NAME="A0"; else NAME="${CELL}"; fi
  S=${CONFIRM_SEEDS[$(( IDX % NCS ))]}
  EXTRA=""
  for E in "${CELLS[@]}"; do
    [[ "${E%%:*}" == "${NAME}" ]] && EXTRA="${E#*:}"
  done
  echo "[sw2-confirm] cell=${NAME} seed=${S} extra='${EXTRA}'"
  python3 -u -m automl.topo.train ${BASE} --seed "${S}" \
      --tag "sw2_${NAME}_s${S}" ${EXTRA} --out-dir "${OUT}"
  echo "SW2 DONE cell=${NAME} seed=${S} $(date -Is)"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID}
CELL_I=$(( IDX / NS ))
S=${SEEDS[$(( IDX % NS ))]}
ENTRY="${CELLS[${CELL_I}]}"
NAME="${ENTRY%%:*}"
EXTRA="${ENTRY#*:}"

echo "[sw2] cell=${NAME} seed=${S} extra='${EXTRA}'"
python3 -u -m automl.topo.train ${BASE} --seed "${S}" \
    --tag "sw2_${NAME}_s${S}" ${EXTRA} --out-dir "${OUT}"
echo "SW2 DONE cell=${NAME} seed=${S} $(date -Is)"
