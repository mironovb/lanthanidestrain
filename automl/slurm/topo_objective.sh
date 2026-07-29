#!/bin/bash
#SBATCH --job-name=ln_obj
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/obj_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/obj_%A_%a.err
#
# The decomposed objective: stop spending the gradient on the block mean.
#
# Measured on this dataset, the composition-block mean carries Var 2.41 and the
# within-block contrast Var 0.25 under the strict key -- so the published Huber
# objective puts ~91% of its gradient on a quantity the adjacent-pair metric
# never reads, and which CatBoost already predicts better than any net here
# (overall R2 +0.4987 against the stack's +0.4369).
#
# --pair-loss-weight could only ever ADD a contrast term on top of the full MSE.
# It had no way to take the level term away.  --level-weight splits the loss
# into a per-block level term and the contrast term so the two can be weighted
# independently rather than by whatever ratio their variances happen to have.
#
#   sbatch --array=0-47 automl/slurm/topo_objective.sh
#
# 6 cells x 8 seeds.  Sized for the real cluster limit: GrpTRES caps this
# account at ONE node on xeon-g6-volta, i.e. 2 concurrent jobs, so 48 runs is
# about three hours of wall clock and a 648-cell grid would be a fortnight.
#
# Selection protocol (PI_SWEEP amendment 2a, reused deliberately): every run
# trains on ALL 162 extractants and the tune/confirm split is applied at
# SCORING time.  --restrict-groups was tried in the persistence-image sweep and
# removed 57% of the training rows, collapsing the arm from +0.156 to +0.036 and
# leaving the selection rule unable to rank anything.  That cost 66 GPU runs and
# is not repeated here.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_objective"
mkdir -p "${OUT}"

SEEDS=(7 11 23 37 42 51 67 83)
NS=${#SEEDS[@]}

# 6 cells: level weight x blocking.  level_weight 1.0 with the binned key is the
# closest cell to the published objective and acts as the internal anchor.
LEVELS=(0.1 0.3 1.0)
KEYS=(composition_key strict_composition_key)

CELL=$(( SLURM_ARRAY_TASK_ID / NS ))
S=${SEEDS[$(( SLURM_ARRAY_TASK_ID % NS ))]}
LW=${LEVELS[$(( CELL % 3 ))]}
BK=${KEYS[$(( CELL / 3 ))]}
SHORT=$([ "${BK}" = "composition_key" ] && echo bin || echo str)

TAG="obj_lw${LW}_${SHORT}_s${S}"
echo "[obj] cell=${CELL} level_weight=${LW} block_key=${BK} seed=${S}"
python3 -u -m automl.topo.train --arch snn --tag "${TAG}" \
    --pair-loss-weight 2.0 --select-on adjacent \
    --level-weight "${LW}" --block-key "${BK}" \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "OBJ DONE cell=${CELL} lw=${LW} key=${BK} seed=${S} $(date -Is)"
