#!/bin/bash
#SBATCH --job-name=ln_encoder
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/enc_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/enc_%A_%a.err
#
# Is it *simplicial*, or merely *3D message passing*?
# Pre-registered in automl/reports/ENCODER_PREREGISTRATION.md, committed at
# 6abaf35 before either arm had ever been run.
#
#   ARM=G0 sbatch --array=0-15 automl/slurm/topo_encoder.sh   # --no-triangles
#   ARM=D0 sbatch --array=0-15 automl/slurm/topo_encoder.sh   # --arch dist
#
# Both arms use the published S0 configuration in every other respect --
# contrast objective, adjacent-pair checkpoint selection, 5 folds x 3 repeats,
# and the same 16 seeds -- so the inner-validation split and batch order match
# S0 exactly and the paired bootstrap compares arms rather than splits.
#
# NOTE (amendment 1 to the pre-registration): --deterministic is deliberately
# NOT used here.  The published S0 ensemble these arms are compared against was
# trained without it, and an arm trained under a different reduction order is
# not matched to it.  Determinism is measured separately in determinism.sh; the
# two questions are kept apart rather than confounded.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_encoder"
mkdir -p "${OUT}"

# The exact 16 seeds behind the published SNN ensemble.
SEEDS=(7 11 23 37 42 51 67 83 211 223 233 241 251 263 271 281)
S=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
ARM="${ARM:?set ARM to G0 or D0}"

CONTRAST="--pair-loss-weight 2.0 --select-on adjacent"
case "${ARM}" in
  G0) ARCH=snn;  TAG="g0_notri_s${S}"; EXTRA="--no-triangles" ;;
  D0) ARCH=dist; TAG="d0_dist_s${S}";  EXTRA="" ;;
  *)  echo "unknown ARM=${ARM}" >&2; exit 1 ;;
esac

echo "[enc] arm=${ARM} arch=${ARCH} seed=${S} extra='${EXTRA}'"
python3 -u -m automl.topo.train --arch "${ARCH}" --tag "${TAG}" \
    ${CONTRAST} ${EXTRA} \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "ENC DONE arm=${ARM} seed=${S} $(date -Is)"
