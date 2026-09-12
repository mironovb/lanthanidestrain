#!/bin/bash
#SBATCH --job-name=tf_smoke
#SBATCH --partition=debug-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_smoke_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_smoke_%j.err
# GPU smoke tests for the three transformer experiments: they must run end
# to end on the real data before any seed campaign is queued.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_tf_smoke"; mkdir -p "${OUT}"

echo "=== 1. attention encoder, dense, --smoke ==="
python3 -u -m automl.topo.train --arch attn --preset baseline_2d \
  --filtration-max 4.0 --heavy-only --pair-loss-weight 4.0 --rbf-bins 64 \
  --select-on adjacent --folds 5 --repeats 1 --seed 42 --deterministic \
  --smoke --tag smoke_attn --out-dir "${OUT}" || echo "SMOKE 1 FAILED"
echo "=== 2. attention encoder, sparse, --smoke ==="
python3 -u -m automl.topo.train --arch attn --attn-sparse --preset baseline_2d \
  --filtration-max 4.0 --heavy-only --pair-loss-weight 4.0 --rbf-bins 64 \
  --select-on adjacent --folds 5 --repeats 1 --seed 42 --deterministic \
  --smoke --tag smoke_attnsp --out-dir "${OUT}" || echo "SMOKE 2 FAILED"
echo "=== 3. FT-Transformer, 1 seed, 1 repeat, 6 epochs ==="
python3 -u -m automl.topo.anchored_ft --cells ft_anch --seeds 1 --repeats 1 \
  --epochs 6 || echo "SMOKE 3 FAILED"
echo "=== 4. block transformer, 1 seed, 1 repeat, 6 epochs ==="
python3 -u -m automl.topo.block_transformer --seeds 1 --repeats 1 --epochs 6 \
  || echo "SMOKE 4 FAILED"
echo "TF_SMOKE DONE $(date -Is)"
