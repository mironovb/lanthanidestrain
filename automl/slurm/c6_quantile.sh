#!/bin/bash
#SBATCH --job-name=ln_c6q
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6q_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6q_%A_%a.err
# H2 MEDIAN TEST.  Quantile:alpha=0.5 IS median regression and should MATCH MAE.
# alpha 0.3 / 0.7 are equally "robust" but target a different quantile; if they
# lose, the mechanism is the MEDIAN specifically, not robustness or L1 curvature.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh; module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
V=(q50 q30 q70)
ONLY=${V[${SLURM_ARRAY_TASK_ID}]}
python3 -u -m automl.topo.c6_partners --which catboost --only "${ONLY}" \
    --seeds 8 --repeats 3 --restrict screen_select
echo "C6Q DONE ${ONLY} rc=$? $(date -Is)"
