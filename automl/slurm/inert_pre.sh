#!/bin/bash
#SBATCH --job-name=inert_pre
#SBATCH --partition=debug-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/inert_pre_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/inert_pre_%j.err
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
python3 -u -m automl.topo.train --arch dist --preset baseline_2d \
  --filtration-max 4.0 --heavy-only --pair-loss-weight 4.0 --rbf-bins 64 \
  --select-on adjacent --epochs 60 --folds 5 --repeats 1 --seed 201 \
  --deterministic --tag "${TAG:?}" --out-dir automl/artifacts/topo_inert
echo "INERT ${TAG} DONE $(date -Is)"
