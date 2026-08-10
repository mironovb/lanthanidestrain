#!/bin/bash
#SBATCH --job-name=c9smoke
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c9smoke_%j.out
# Validate --pair-head / --pair-reconcile before spending 24 cells on them.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"
python3 -u -m automl.topo.train --arch dist --pair-loss-weight 2.0 \
    --select-on adjacent --filtration-max 4.0 --rbf-bins 64 \
    --folds 5 --repeats 1 --deterministic --pair-head --pair-reconcile \
    --seed 7 --tag c9smoke --out-dir "${REPO}/automl/artifacts/topo_c9"
echo "C9SMOKE_RC=$?"
