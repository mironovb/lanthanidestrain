#!/bin/bash
#SBATCH --job-name=ln_c6pn
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6pn_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6pn_%A_%a.err
# Overnight CPU work: quantile-loss variants of the OTHER tabular families, so
# the stack gains arms that are strong AND decorrelated rather than more of the
# same.  CatBoost-MAE already displaced the fingerprint net entirely.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh; module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
V=(q70 q60 mae_rsm03 mae_slow q65 q75)
ONLY=${V[${SLURM_ARRAY_TASK_ID}]}
python3 -u -m automl.topo.c6_partners --which catboost --only "${ONLY}" \
    --seeds 16 --repeats 3 --restrict full
echo "C6PN DONE ${ONLY} rc=$? $(date -Is)"
