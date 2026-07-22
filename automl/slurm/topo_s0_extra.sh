#!/bin/bash
#SBATCH --job-name=ln_s0x
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/s0x_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/s0x_%A_%a.err
#
# More seeds of the UNCHANGED S0 configuration.
#
# S2 failed because it changed the winning config -- conformers, block-centring
# and pretraining each hurt (the ablation is unambiguous). Running more seeds of
# the SAME config is a different thing entirely: it is not a new arm, it is a
# better estimate of the ensemble S0 already defines, and S0's ensemble gain
# (+0.060 over its seed mean) had not visibly flattened at 16 seeds.
#
# S0 = +0.2382 (16 seeds) against the repaired baseline's +0.2206; the gap is
# +0.0261 [-0.005, +0.076], n.s. A converged ensemble is the cleanest remaining
# shot at that pre-registered claim, and it changes nothing about the model.
#
# Output goes to a SEPARATE directory so the published 16-seed S0 in
# topo_adj_seeds / topo_adjacent is untouched and control_factorial keeps
# reproducing +0.2382 exactly.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_s0_extra"
mkdir -p "${OUT}"

# 32 fresh seeds, disjoint from the published 16 (7,11,23,37,42,51,67,83,
# 211,223,233,241,251,263,271,281) and from S2's extension set.
SEEDS=(401 409 419 421 431 433 439 443 449 457 461 463 467 479 487 491
       499 503 509 521 523 541 547 557 563 569 571 577 587 593 599 601)
S=${SEEDS[${SLURM_ARRAY_TASK_ID}]}

echo "[s0x] seed=${S} (unchanged S0 config)"
python3 -m automl.topo.train --arch snn --tag "s0x_s${S}" \
    --pair-loss-weight 2.0 --select-on adjacent \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "S0X DONE seed=${S} $(date -Is)"
