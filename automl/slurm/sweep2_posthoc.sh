#!/bin/bash
#SBATCH --job-name=ln_sw2ph
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/sw2ph_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/sw2ph_%A_%a.err
#
# POST-HOC mechanism test for sweep2 cell A1.  NOT in SWEEP2_PREREGISTRATION.md
# and claiming no confirmatory status -- it explains a result, it does not
# establish one.
#
# A1 adds 119 angular/polyhedral columns to the tabular head and the
# adjacent-pair metric collapses (-0.3167 on the tune half) while overall R2
# gives up only 0.10.  87 of those columns vary WITHIN a composition block but
# correlate with dy at a median |r| of only 0.049, so the hypothesis is that the
# head fits within-block geometry variation that the metric cannot use.
#
# This cell keeps the same 119 columns and the same between-block content, and
# removes the within-block variation by replacing each column with its per-block
# mean.  Leak-free: 0 of 552 blocks span more than one extractant group, so the
# mean never crosses a CV fold.
#
#   Predicted if the mechanism holds : A1BM lands near the A0 anchor
#   Predicted if it does not         : A1BM stays down near A1
#
# Only 4 runs: A0 and A1 already exist at 4 seeds from the sweep itself.
#
#   automl/slurm/campaign_driver.sh automl/slurm/sweep2_posthoc.sh 4 4 34
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_sweep2"
mkdir -p "${OUT}"

SEEDS=(7 11 23 37)
S=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
BASE="--arch snn --no-triangles --pair-loss-weight 2.0 --select-on adjacent --deterministic --folds 5 --repeats 3"

echo "[sw2ph] cell=A1BM seed=${S}  (A1 columns, within-block variation removed)"
python3 -u -m automl.topo.train ${BASE} --seed "${S}" \
    --preset baseline_2d_shape --extra-block-mean \
    --tag "sw2_A1BM_s${S}" --out-dir "${OUT}"
echo "SW2PH DONE seed=${S} $(date -Is)"
