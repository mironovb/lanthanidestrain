#!/bin/bash
#SBATCH --job-name=ln_c6qf
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6qf_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6qf_%A_%a.err
# The alpha sweep peaked at 0.70 with a smooth interior maximum.  Confirm the
# peak and one neighbour on FULL data at 16 seeds so they can enter the stack
# and be scored on the held-out third.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh; module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
V=(q70 q60)
ONLY=${V[${SLURM_ARRAY_TASK_ID}]}
python3 -u -m automl.topo.c6_partners --which catboost --only "${ONLY}" \
    --seeds 16 --repeats 3 --restrict full
echo "C6QF DONE ${ONLY} rc=$? $(date -Is)"
