#!/bin/bash
#SBATCH --job-name=c17smoke
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c17smoke_%j.out
# snn has never run with these flags. Validate that the loss-side flags and the
# mphys preset are actually honoured on the simplicial encoder BEFORE spending
# 144 cells -- the --pair-head lesson was that a flag can be accepted, recorded
# as enabled, and silently ignored by the architecture.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_c17"
echo "### combo arm (all three levers at once, the most likely to break) ###"
python3 -u -m automl.topo.train --arch snn --select-on adjacent \
    --filtration-max 3.5 --folds 5 --repeats 1 --deterministic \
    --pair-loss-weight 4.0 --pair-adj-weight 10.0 --preset baseline_2d_mphys \
    --seed 501 --tag c17smoke_combo --out-dir "${OUT}"
echo "C17SMOKE_RC=$?"
