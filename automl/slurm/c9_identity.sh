#!/bin/bash
#SBATCH --job-name=c9ident
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c9ident_%j.out
# The pair_head change touches DistanceNet's __init__, which every published
# --arch dist run used.  With --pair-head OFF the model must be bit-identical,
# or the edit silently invalidates the whole study.  Two runs, same seed,
# --deterministic: the OOF predictions must match to 0.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_c9"
python3 -u -m automl.topo.train --arch dist --pair-loss-weight 2.0 \
    --select-on adjacent --filtration-max 4.0 --rbf-bins 64 \
    --folds 5 --repeats 1 --deterministic --seed 7 \
    --tag c10ident_film --out-dir "${OUT}"
echo "C9IDENT_RC=$?"
