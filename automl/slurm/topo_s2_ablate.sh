#!/bin/bash
#SBATCH --job-name=ln_s2ab
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/s2ab_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/s2ab_%A_%a.err
#
# Leave-one-out ablation of the S2 levers.
#
# DESCRIPTIVE ONLY. The pre-registration bundles the four levers into a single
# confirmatory arm and explicitly trades attribution-of-cause for power; nothing
# here is a confirmatory test and none of it may be substituted for the primary
# endpoint. It exists to answer "which lever did the work", which is the
# question the bundling gave up -- and to catch the case where one lever helps
# while another hurts, which a bundled arm cannot show.
#
# Leave-one-out rather than one-at-a-time: the levers are meant to compose, and
# what a reader wants to know is what would be lost by dropping each.
#
# 8 seeds per cell, drawn from the S2 matched set so every ablation stays
# seed-paired with the full arm.
#
#   CELL=noconf sbatch --array=0-7 automl/slurm/topo_s2_ablate.sh
#
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_s2_ablate"
mkdir -p "${OUT}"

SEEDS=(7 11 23 37 42 51 67 83)
S=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
CELL="${CELL:?set CELL to one of noconf nocentre nopretrain}"

BASE="--pair-loss-weight 2.0 --select-on adjacent"
case "${CELL}" in
  noconf)     CFG="${BASE} --block-centre --pretrain-epochs 20 --conformers 1" ;;
  nocentre)   CFG="${BASE} --conformers 3 --pretrain-epochs 20" ;;
  nopretrain) CFG="${BASE} --conformers 3 --block-centre --pretrain-epochs 0" ;;
  *) echo "unknown CELL=${CELL}"; exit 1 ;;
esac

echo "[s2ab] cell=${CELL} seed=${S} cfg='${CFG}'"
python3 -m automl.topo.train --arch snn --tag "s2ab_${CELL}_s${S}" ${CFG} \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "S2AB DONE cell=${CELL} seed=${S} $(date -Is)"
