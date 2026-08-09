#!/bin/bash
#SBATCH --job-name=ln_c6p
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6p_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6p_%A_%a.err
#
# Re-tune the two non-3D stack partners on the ADJACENT-PAIR metric.
#
# CPU only, on xeon-p8, so it does not compete with the GPU waves for the one
# xeon-g6-volta node the account may hold.
#
#   sbatch --array=0-1 automl/slurm/c6_partners.sh
#     task 0 -> fingerprint network, task 1 -> CatBoost
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) WHICH=fcnn ;;
  1) WHICH=catboost ;;
  *) echo "unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

echo "[c6p] ${WHICH} on ${RESTRICT:-screen_select}"
python3 -u -m automl.topo.c6_partners --which "${WHICH}" \
    --seeds "${NSEEDS:-8}" --repeats "${REPEATS:-3}" \
    --restrict "${RESTRICT:-screen_select}"
echo "C6P DONE ${WHICH} rc=$? $(date -Is)"
