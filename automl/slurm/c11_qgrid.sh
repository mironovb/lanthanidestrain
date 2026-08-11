#!/bin/bash
#SBATCH --job-name=c11_qgrid
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c11_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c11_%A_%a.err
#
# Re-grid the CatBoost hyperparameters around the loss that actually wins.
#
# C6 located the tabular optimum at Quantile(alpha=0.6): +0.2579 against q65
# +0.2430 and q70 +0.2321. But depth/lr/l2/rsm were re-gridded around MAE
# (the mae_* block) and never around q60 -- exactly the gap that block was
# added to close for MAE, left open for its successor. The tabular arm is the
# largest single contributor in the project (MAE alone was +0.1066 adjacent),
# so this is the highest-value CPU work available.
#
# One grid entry per array task, so the 8-node allocation is used in parallel
# and a slow cell cannot hold up the rest.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
ENTRIES=(q60_deep q60_shallow q60_slow q60_rsm03 q60_l2_10 q60_rsm03_deep q60_slow_deep q55)
E="${ENTRIES[${SLURM_ARRAY_TASK_ID}]}"
echo "[c11] catboost grid entry ${E}"
python3 -u -m automl.topo.c6_partners --which catboost --only "${E}" \
    --seeds "${NSEEDS:-8}" --repeats "${REPEATS:-3}" \
    --restrict "${RESTRICT:-screen_select}"
echo "C11 DONE ${E} rc=$? $(date -Is)"
