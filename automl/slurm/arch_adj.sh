#!/bin/bash
#SBATCH --job-name=arch_adj
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/arch_adj_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/arch_adj_%j.err
#
# A1 of the post-0.313 campaign: score the anchored / two-stage / pairwise-delta
# architecture family on sel_adj_logSF_r2 for the first time.  The 457-run arch
# sweep predates the adjacent metric (automl/reports/all_results.csv has no
# sel_adj_* column); automl.evaluation.full_metrics now emits it, so a re-run
# is the measurement.  ok_only rows = the legacy iteration population; OOF
# parquets saved so surviving arms can enter the pair-fitted stack directly.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

OUT="${REPO}/automl/artifacts/arch_adj"
mkdir -p "${OUT}"

for BASE in catboost lgbm; do
  echo "[arch_adj] base=${BASE} $(date -Is)"
  python3 -u -m automl.run_sweep --mode arch \
      --out-dir "${OUT}" \
      --models "${BASE}" \
      --presets baseline_2d,selectivity \
      --row-filters ok_only \
      --n-splits 5 --repeats 2 --seed 42 \
      --n-jobs 12 --save-oof
done
echo "ARCH_ADJ DONE $(date -Is)"
