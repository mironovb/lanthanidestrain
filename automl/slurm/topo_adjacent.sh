#!/bin/bash
#SBATCH --job-name=ln_adj
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/adj_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/adj_%A_%a.err
# Targeting the adjacent-lanthanide-pair claim directly.
#
# Every arm so far optimised absolute log D, while the adjacent-pair metric
# scores predicted *differences* between two lanthanides sharing an extractant
# and conditions.  A model can fit absolute log D well while its within-block
# contrasts are noise -- which is exactly the pattern observed.
#
# Two new levers, both legal under leave-extractants-out:
#   --pair-loss-weight  auxiliary loss on within-composition pairwise
#                       differences, weighted 3x towards |dZ| = 1 neighbours;
#                       batches are whole composition blocks so pairs exist
#   --select-on adjacent  early stopping on the adjacent-pair R2 of the INNER
#                       validation split (held-out extractants, never test)
#
# pi_hybrid is included because it already scored the best adjacent-pair R2 of
# any arm (+0.156, above CatBoost's +0.142) under the plain objective.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_adjacent"

case "${SLURM_ARRAY_TASK_ID}" in
  0) A="--arch picnn --tag pi_pair0.5  --pair-loss-weight 0.5" ;;
  1) A="--arch picnn --tag pi_pair2    --pair-loss-weight 2.0" ;;
  2) A="--arch picnn --tag pi_pair2_sel --pair-loss-weight 2.0 --select-on adjacent" ;;
  3) A="--arch picnn --tag pi_sel      --select-on adjacent" ;;
  4) A="--arch snn   --tag snn_pair2   --pair-loss-weight 2.0" ;;
  5) A="--arch snn   --tag snn_pair2_sel --pair-loss-weight 2.0 --select-on adjacent" ;;
  6) A="--arch snn   --tag snn_pair5_sel --pair-loss-weight 5.0 --select-on adjacent" ;;
  7) A="--arch snn   --tag snn_wide_pair --dim 160 --layers 4 --dropout 0.2 --pair-loss-weight 2.0 --select-on adjacent" ;;
  8) A="--arch picnn --tag pi_pair5_sel --pair-loss-weight 5.0 --select-on adjacent" ;;
  9) A="--arch snn   --tag snn_pair2_sel_s7 --pair-loss-weight 2.0 --select-on adjacent --seed 7" ;;
  *) echo unknown; exit 1 ;;
esac

echo "[adj] task ${SLURM_ARRAY_TASK_ID}: ${A}"
python3 -m automl.topo.train ${A} --folds 5 --repeats 3 --out-dir "${OUT}"
echo "ADJ DONE task=${SLURM_ARRAY_TASK_ID} $(date -Is)"
