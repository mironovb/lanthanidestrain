#!/bin/bash
#SBATCH --job-name=ln_tbase
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tbase_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tbase_%A_%a.err
# Stage 5b: the two baselines the topological arms are measured against.
#
# Both are reported, deliberately.  The abstract benchmarks against an FCNN on
# ECFP + RDKit, so that one has to be here or the comparison is not the one
# claimed.  But the prior study found CatBoost + inverse-extractant weighting is
# far stronger, and beating only the weak baseline would be a strawman -- so the
# strong one is here too, and any win is reported against both.
#
# row_filter=ok_only is exactly the 4,746-row set the SNN arms train on
# (verified identical), so paired_bootstrap pairs on the same rows and the same
# leave-extractants-out folds (5 splits x 3 repeats, seed 42).
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) M=catboost; W=group_inv ;;   # the strong baseline
  1) M=catboost; W=none ;;
  2) M=mlp;      W=none ;;        # the abstract's own FCNN baseline
  3) M=mlp;      W=group_inv ;;
esac

echo "[tbase] model=$M weights=$W"
python3 -m automl.run_sweep \
  --mode models \
  --out-dir "${REPO}/automl/artifacts/sweeps/topo_baselines" \
  --presets baseline_2d \
  --models "$M" --weight-schemes "$W" \
  --row-filters ok_only \
  --n-splits 5 --repeats 3 --seed 42 \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --save-oof
echo "TBASE DONE model=$M weights=$W $(date -Is)"
