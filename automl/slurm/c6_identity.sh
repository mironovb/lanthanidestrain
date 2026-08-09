#!/bin/bash
#SBATCH --job-name=ln_c6id
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6id_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6id_%A_%a.err
#
# The default-off gate.  Runs ONE short deterministic configuration twice --
# once from a pristine worktree at HEAD, once from the working tree -- and the
# two OOF parquets must agree to the last bit.
#
# Why --deterministic is not optional here.  The published arms were trained
# without it and carry a ~0.009 run-to-run floor caused by index_add_ scatter
# atomics (DETERMINISM_RESULTS.md).  Comparing a new run against a published
# one therefore measures GPU reduction order, not code: the first attempt at
# this gate returned max|d| = 1.9 and proved nothing.  Two deterministic runs
# of the same code agree exactly, so any difference IS the code.
#
#   BASE=<worktree> sbatch --array=0-1 automl/slurm/c6_identity.sh
#     task 0 -> HEAD, task 1 -> working tree
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
BASE="${BASE:?set BASE=<path to pristine HEAD worktree>}"
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONWARNINGS=ignore CUBLAS_WORKSPACE_CONFIG=":4096:8"

OUT="${REPO}/automl/artifacts/c6_identity"
mkdir -p "${OUT}"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  ROOT="${BASE}"; TAG="idbase"
else
  ROOT="${REPO}"; TAG="idnew"
fi
export PYTHONPATH="${ROOT}"
cd "${ROOT}"
echo "[id] task ${SLURM_ARRAY_TASK_ID} root=${ROOT} tag=${TAG}"

# Short but exercising the real paths: contrast objective on, adjacent-pair
# checkpoint selection on, whole-block batching on.  8 epochs is enough for the
# loss to have been evaluated hundreds of times, which is what is being tested.
python3 -u -m automl.topo.train --arch dist \
    --pair-loss-weight 2.0 --select-on adjacent \
    --folds 5 --repeats 1 --epochs 8 --seed 7 --deterministic \
    --tag "${TAG}" --out-dir "${OUT}"
echo "ID DONE ${TAG} rc=$? $(date -Is)"
