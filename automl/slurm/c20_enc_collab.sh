#!/bin/bash
#SBATCH --job-name=c20_enc
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c20_enc_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c20_enc_%j.err
# C20: distance encoder on the collaborator's expanded population.
# Submit only after inert_check passes.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_c20"
mkdir -p "${OUT}"
for SEED in ${SEEDS:-42 51 67 83}; do
  python3 -u -m automl.topo.train --arch dist --preset baseline_2d \
    --filtration-max 4.0 --heavy-only --pair-loss-weight 4.0 --rbf-bins 64 \
    --select-on adjacent --epochs 60 --folds 5 --repeats 3 --seed "${SEED}" \
    --deterministic --edge-asset collab --population collab \
    --tag "c20_plw4col_s${SEED}" --out-dir "${OUT}"
done
echo "C20_ENC DONE $(date -Is)"
