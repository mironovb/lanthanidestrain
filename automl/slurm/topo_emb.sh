#!/bin/bash
#SBATCH --job-name=ln_emb
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/emb_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/emb_%A_%a.err
#
# Out-of-fold SNN embeddings, so the learned topological REPRESENTATION can be
# handed to CatBoost instead of to the topo harness's own MLP head.
#
# Motivation, measured: T0w (harness MLP on tabular, contrast objective, 16
# seeds) scores +0.2006 while the repaired sklearn MLP on the SAME features
# scores +0.2206. The harness head is the weaker part, so every end-to-end
# topological arm has been scoring encoder+head together and may be
# under-selling the encoder.
#
# Array 0-3: seeds 42, 7, 11, 23 for the trained encoder (tag embsnn).
# Array 4  : the RANDOM-init control (tag embrand) -- 0 training epochs, so the
#            encoder is never updated and its embedding is a fixed random
#            projection of the same geometry. If trained ~= random, the encoder
#            learned nothing worth having.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_emb"
mkdir -p "${OUT}"

T=${SLURM_ARRAY_TASK_ID}
SEEDS=(42 7 11 23)
if [ "$T" -lt 4 ]; then
  S=${SEEDS[$T]}
  TAG="embsnn_s${S}"
  EXTRA="--pair-loss-weight 2.0 --select-on adjacent --epochs 60"
else
  S=42
  TAG="embrand_s${S}"
  # epochs 0 -> the encoder keeps its initialisation; the embedding is a random
  # projection of the geometry through the same architecture.
  EXTRA="--pair-loss-weight 2.0 --select-on adjacent --epochs 0"
fi

echo "[emb] task=${T} tag=${TAG} seed=${S} ${EXTRA}"
python3 -m automl.topo.train --arch snn --tag "${TAG}" ${EXTRA} \
    --dump-embeddings --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "EMB DONE tag=${TAG} $(date -Is)"
