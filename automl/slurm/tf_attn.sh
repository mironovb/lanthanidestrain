#!/bin/bash
#SBATCH --job-name=tf_attn
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_attn_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_attn_%j.err
#
# Attention-based 3D encoder (--arch attn), same configuration as the
# c15_plw4 distance encoder it is compared with: baseline_2d, 4.0 A, heavy
# atoms, contrast weight 4, 64 RBF bins, adjacent selection, 5 folds x 3
# repeats, deterministic.  Per-seed tags (the lesson of C19).
#   SEEDS   seeds to run (default 8)
#   POP     ok_only (default) | has3d  -- has3d adds --edge-asset has3d
#   SPARSE  1 -> --attn-sparse (the <= cutoff control)
#   OUTDIR  artefact dir (default topo_tf_attn)
#   EXTRA_ARGS  appended verbatim to every train.py call (e.g. "--lr 1e-3")
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
POP="${POP:-ok_only}"; SPARSE="${SPARSE:-0}"
OUT="${REPO}/automl/artifacts/${OUTDIR:-topo_tf_attn}"; mkdir -p "${OUT}"
EXTRA=""
[ "${POP}" = "has3d" ] && EXTRA="--edge-asset has3d --population has3d"
TAGBASE="tfattn"
[ "${SPARSE}" = "1" ] && { EXTRA="${EXTRA} --attn-sparse"; TAGBASE="tfattnsp"; }
[ "${POP}" = "has3d" ] && TAGBASE="${TAGBASE}h3d"
for SEED in ${SEEDS:-42 51 67 83 91 103 107 109}; do
  python3 -u -m automl.topo.train --arch attn --preset baseline_2d \
    --filtration-max 4.0 --heavy-only --pair-loss-weight 4.0 --rbf-bins 64 \
    --select-on adjacent --epochs 60 --folds 5 --repeats 3 --seed "${SEED}" \
    --deterministic ${EXTRA} ${EXTRA_ARGS:-} \
    --tag "${TAGBASE}_s${SEED}" --out-dir "${OUT}"
done
echo "TF_ATTN DONE pop=${POP} sparse=${SPARSE} extra=[${EXTRA_ARGS:-}] $(date -Is)"
