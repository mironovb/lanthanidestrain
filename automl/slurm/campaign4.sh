#!/bin/bash
#SBATCH --job-name=ln_c4
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c4_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c4_%A_%a.err
#
# Pre-registered in automl/reports/CAMPAIGN4_PREREGISTRATION.md (+ Amendments 1, 2).
#
#   automl/slurm/campaign_driver.sh automl/slurm/campaign4.sh 12 8 30
#   automl/slurm/campaign_driver.sh automl/slurm/campaign4.sh 24 8 30 MODE=confirm CELL=neutral
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_c4"; mkdir -p "${OUT}"

SEEDS=(7 11 23 37)
CONFIRM_SEEDS=(101 103 107 109 113 127 131 137 139 149 151 157)
NS=${#SEEDS[@]}
# Identical to the sweep2/campaign3 anchor, so the only thing that varies across
# the three arms is which geometry set the encoder sees.
BASE="--arch snn --no-triangles --pair-loss-weight 2.0 --select-on adjacent --deterministic --folds 5 --repeats 3"
ARMS=(shipped control neutral)

if [[ "${MODE:-run}" == "confirm" ]]; then
  : "${CELL:?MODE=confirm needs CELL=<arm>}"
  NCS=${#CONFIRM_SEEDS[@]}
  IDX=${SLURM_ARRAY_TASK_ID}
  if (( IDX < NCS )); then G="control"; else G="${CELL}"; fi
  S=${CONFIRM_SEEDS[$(( IDX % NCS ))]}
  echo "[c4-confirm] geometry=${G} seed=${S}"
  python3 -u -m automl.topo.train ${BASE} --geometry "${G}" --seed "${S}" \
      --tag "c4_${G}_s${S}" --out-dir "${OUT}"
  echo "C4 DONE ${G} ${S} $(date -Is)"
  exit 0
fi

IDX=${SLURM_ARRAY_TASK_ID}
G=${ARMS[$(( IDX / NS ))]}
S=${SEEDS[$(( IDX % NS ))]}
echo "[c4] geometry=${G} seed=${S}"
python3 -u -m automl.topo.train ${BASE} --geometry "${G}" --seed "${S}" \
    --tag "c4_${G}_s${S}" --out-dir "${OUT}"
echo "C4 DONE ${G} ${S} $(date -Is)"
