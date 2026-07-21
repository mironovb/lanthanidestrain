#!/bin/bash
#SBATCH --job-name=ln_topocv
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/topocv_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/topocv_%A_%a.err
# Stage 3c/4: leave-extractants-out grouped CV for the topological arms.
#
# Every arm uses the SAME folds (automl.evaluation.grouped_folds, seed 42) and
# writes OOF predictions in the tabular-sweep schema, so any pair of arms --
# including the CatBoost and FCNN baselines -- can go through
# automl.compare.paired_bootstrap on identical rows.
#
# The arms are chosen so each conclusion has its own control:
#   0,1  hybrid SNN, with and without self-supervised pretraining
#   2    topology-only SNN     -> does the simplicial encoder carry signal alone
#   3    denser filtration     -> is 3.5 A throwing away structure
#   4    all-atom (with H)     -> does heavy-atom sparsification cost anything
#   5    wider/deeper encoder  -> capacity control
#   6,7  PI-CNN, hybrid and topology-only (the corrected persistence-image test)
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARGS="--arch snn --tag snn_hybrid" ;;
  1) ARGS="--arch snn --tag snn_pretrain --pretrain-epochs 30" ;;
  2) ARGS="--arch snn --tag snn_topoonly --topology-only" ;;
  3) ARGS="--arch snn --tag snn_filt5 --filtration-max 5.0" ;;
  4) ARGS="--arch snn --tag snn_allatom --all-atoms --filtration-max 3.0" ;;
  5) ARGS="--arch snn --tag snn_wide --dim 160 --layers 4 --dropout 0.2" ;;
  6) ARGS="--arch picnn --tag pi_hybrid" ;;
  7) ARGS="--arch picnn --tag pi_topoonly --topology-only" ;;
  *) echo "unknown task"; exit 1 ;;
esac

echo "[topocv] task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
python3 -m automl.topo.train ${ARGS} --folds 5 --repeats 3 --seed 42
echo "TOPO CV DONE task=${SLURM_ARRAY_TASK_ID} $(date -Is)"
