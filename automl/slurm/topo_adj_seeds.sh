#!/bin/bash
#SBATCH --job-name=ln_adjs
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/adjs_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/adjs_%A_%a.err
# Seed replicates of the two best adjacent-pair configurations, for ensembling.
#
# Why this is necessary rather than optional: the paired bootstrap on the
# adjacent-pair metric returns intervals ~0.175 wide.  pi_hybrid's apparent
# +0.014 edge over CatBoost came back as delta = -0.004 [-0.111, +0.064] --
# indistinguishable.  Any single run's point estimate on this metric is mostly
# noise, so a claim needs either a much larger effect or less variance.
#
# Averaging out-of-fold predictions over seeds attacks the variance directly and
# is not metric-gaming: every seed uses the same folds, the ensemble is formed
# without consulting the test metric, and the result is still scored by the same
# paired cluster bootstrap.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_adj_seeds"

T=${SLURM_ARRAY_TASK_ID}
SEEDS=(11 23 37 51 67 83)
if [ "$T" -lt 6 ]; then
  ARCH=picnn; CFG="--pair-loss-weight 2.0 --select-on adjacent"; S=${SEEDS[$T]}
  TAG="pi_pair2_sel_s${S}"
else
  ARCH=snn;   CFG="--pair-loss-weight 2.0 --select-on adjacent"; S=${SEEDS[$((T-6))]}
  TAG="snn_pair2_sel_s${S}"
fi

echo "[adjs] task ${T}: arch=${ARCH} seed=${S} ${CFG}"
python3 -m automl.topo.train --arch "${ARCH}" --tag "${TAG}" ${CFG} \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "ADJS DONE task=${T} $(date -Is)"
