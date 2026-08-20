#!/bin/bash
#SBATCH --job-name=c11conf
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c11conf_%A_%a.out
# ONE pre-registered look at the report third: q60_rsm03_deep vs q60.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
ENTRIES=(q60_rsm03_deep q60)
E="${ENTRIES[${SLURM_ARRAY_TASK_ID}]}"
python3 -u -m automl.topo.c6_partners --which catboost --only "${E}" \
    --seeds 8 --repeats 3 --restrict report
echo "C11CONF DONE ${E} rc=$? $(date -Is)"
