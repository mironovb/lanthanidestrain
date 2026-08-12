#!/bin/bash
#SBATCH --job-name=c16stack
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c16_%A_%a.out
# The project's headline number is the 3-model STACK (+0.2672), not any single
# arm. C11 improved the CatBoost partner (q60_rsm03_deep, +0.0159 on the
# held-out third). A better component should raise the stack -- but stacking is
# complementarity-driven, so a stronger partner can also correlate more with the
# others and add LESS. That is exactly what the mechanism table in the README
# predicts, and it has never been tested for this partner.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.c6_partners --which catboost --only q60_rsm03_deep \
    --seeds 8 --repeats 3 --restrict full
echo "C16 DONE rc=$? $(date -Is)"
