#!/bin/bash
#SBATCH --job-name=ln_c6pm
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6pm_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6pm_%A_%a.err
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
V=(mae_deep mae_shallow mae_slow mae_rsm03 mae_l2_10 huber huber_d03)
ONLY=${V[${SLURM_ARRAY_TASK_ID}]}
echo "[c6pm] catboost/${ONLY} on screen_select"
python3 -u -m automl.topo.c6_partners --which catboost --only "${ONLY}" \
    --seeds 8 --repeats 3 --restrict screen_select
echo "C6PM DONE ${ONLY} rc=$? $(date -Is)"
