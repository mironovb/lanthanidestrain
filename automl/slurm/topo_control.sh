#!/bin/bash
#SBATCH --job-name=ln_ctrl
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/ctrl_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/ctrl_%A_%a.err
#
# The missing control, as a 2x2 factorial.  Pre-registered in
# automl/reports/CONTROL_PREREGISTRATION.md, committed before this was ever
# submitted.
#
# All 51 runs of the published study used a topological encoder, and the
# mechanism it identified -- "train the contrast, not the absolute value" -- is
# a property of the objective, not of the representation.  So the +0.243 cannot
# be attributed to topology until a tabular model gets the same objective.
# CELL selects which cell of the factorial to run; every cell uses the same 16
# seeds as the published SNN ensemble, so every contrast is matched:
#
#   CELL=T0   tabular  + contrast + adjacent selection   <- the control
#   CELL=T0w  tabular  + contrast + adjacent selection, head_hidden 512
#   CELL=T1   tabular  + plain MSE
#   CELL=P1   picnn    + plain MSE
#   CELL=P0   picnn    + contrast + adjacent selection   (only seed 7 is missing)
#   CELL=S1   snn      + plain MSE
#
# Same partition and GPU as the published runs, so nothing differs numerically
# beyond the arm itself.
#
#   CELL=T0 sbatch --array=0-15 automl/slurm/topo_control.sh
#
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_control"
mkdir -p "${OUT}"

# The exact 16 seeds behind the published SNN ensemble.  Matched seeds mean the
# inner-validation split and batch order are identical across cells, so the
# paired bootstrap compares arms and not splits.
SEEDS=(7 11 23 37 42 51 67 83 211 223 233 241 251 263 271 281)
S=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
CELL="${CELL:?set CELL to one of T0 T0w T1 P1 P0 S1}"

CONTRAST="--pair-loss-weight 2.0 --select-on adjacent"
case "${CELL}" in
  T0)  ARCH=tabular; TAG="tab_pair2_sel_s${S}";   CFG="${CONTRAST}" ;;
  T0w) ARCH=tabular; TAG="tabw_pair2_sel_s${S}";  CFG="${CONTRAST} --head-hidden 512" ;;
  T1)  ARCH=tabular; TAG="tab_plain_s${S}";       CFG="" ;;
  P1)  ARCH=picnn;   TAG="pi_plain_s${S}";        CFG="" ;;
  P0)  ARCH=picnn;   TAG="pi_pair2_sel_ctl_s${S}"; CFG="${CONTRAST}" ;;
  S1)  ARCH=snn;     TAG="snn_plain_s${S}";       CFG="" ;;
  *)   echo "unknown CELL=${CELL}"; exit 1 ;;
esac

echo "[ctrl] cell=${CELL} arch=${ARCH} seed=${S} cfg='${CFG}'"
python3 -m automl.topo.train --arch "${ARCH}" --tag "${TAG}" ${CFG} \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "CTRL DONE cell=${CELL} seed=${S} $(date -Is)"
