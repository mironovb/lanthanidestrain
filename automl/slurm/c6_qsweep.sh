#!/bin/bash
#SBATCH --job-name=ln_c6qs
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6qs_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6qs_%A_%a.err
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh; module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
V=(q60 q65 q75 q80 q85 q90)
ONLY=${V[${SLURM_ARRAY_TASK_ID}]}
python3 -u -m automl.topo.c6_partners --which catboost --only "${ONLY}" \
    --seeds 8 --repeats 3 --restrict "${RESTRICT:-screen_select}"
echo "C6QS DONE ${ONLY} rc=$? $(date -Is)"
