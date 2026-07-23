#!/bin/bash
#SBATCH --job-name=ln_pirep
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pirep_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pirep_%A_%a.err
#
# How precisely can this sweep measure anything?
#
# Stage B re-ran one configuration that Stage A had already run -- same image
# set, same 8 seeds, same code -- because the two manifests share a cell. The
# 8-seed ensemble moved from adjR2 +0.1696 to +0.1587. That is 0.011 of
# run-to-run drift on an identical configuration, against a config-to-config
# range across all of Stage A of only 0.031.
#
# train.py sets torch.manual_seed but never torch.use_deterministic_algorithms
# or the cuDNN determinism flags, so GPU training is not reproducible at fixed
# seed: cuDNN picks algorithms by benchmark and several reductions use
# non-deterministic atomics.
#
# One accidental replicate is not a measurement. This runs three configurations
# spanning the observed range, three independent replicates each, all with the
# identical 8 seeds -- so the only thing varying is the nondeterminism itself.
# The spread across replicates is the noise floor every "difference between
# configurations" in this sweep has to clear.
#
# Writes to its own directory: nothing in the sweep is overwritten.
#
#   sbatch --array=0-8 automl/slurm/pi_replicate.sh
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

# shipped anchor; Stage A's winner; a mid-range configuration.
KEYS=(${PI_REP_KEYS:-13c391a8bb1e3a40 9d6e4c93026dfa0c 6e229d2419e7c99b})
SEEDS=(7 11 23 37 42 51 67 83)

T=${SLURM_ARRAY_TASK_ID}
KEY=${KEYS[$(( T / 3 ))]}
REP=$(( T % 3 ))
IMG="${REPO}/automl/artifacts/pi_sweep/images/img_${KEY}.npz"
OUT="${REPO}/automl/artifacts/pi_replicate/${PI_REP_DIR:-rep}${REP}"
mkdir -p "${OUT}"

export PI_IMAGES_PATH="${IMG}"
echo "[pirep] key=${KEY} replicate=${REP}"

for S in "${SEEDS[@]}"; do
    python3 -m automl.topo.train --arch picnn --tag "rep${REP}_${KEY}_s${S}" \
        --pair-loss-weight 2.0 --select-on adjacent \
        --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}" \
        || echo "PIREP SEED FAILED key=${KEY} rep=${REP} seed=${S}"
done
echo "PIREP DONE key=${KEY} rep=${REP} $(date -Is)"
