#!/bin/bash
#SBATCH --job-name=ln_c6pf
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6pf_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6pf_%A_%a.err
#
# The winning partner variants on ALL 162 extractants, so they can enter the
# endpoint stack.  An arm fitted only on screen+select has no out-of-fold
# prediction on the report third and could not contribute to it.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) WHICH=catboost; ONLY=mae ;;
  1) WHICH=fcnn;     ONLY=narrow ;;
  2) WHICH=fcnn;     ONLY=lr3e3 ;;
  3) WHICH=catboost; ONLY=rsm_03 ;;
  *) echo "unknown task" >&2; exit 1 ;;
esac
echo "[c6pf] ${WHICH}/${ONLY} on full data"
python3 -u -m automl.topo.c6_partners --which "${WHICH}" --only "${ONLY}" \
    --seeds 16 --repeats 3 --restrict full
echo "C6PF DONE ${WHICH}/${ONLY} rc=$? $(date -Is)"
