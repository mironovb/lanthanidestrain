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
