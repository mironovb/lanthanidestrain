#!/bin/bash
#SBATCH --job-name=ln_determ
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/determ_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/determ_%A_%a.err
#
# Does --deterministic actually remove the run-to-run noise?
#
# PI_SWEEP_PRECISION.md measured an 8-seed ensemble moving by 0.0092 between
# identical re-runs and showed that more seeds does not fix it, because part of
# the noise is shared across every seed inside one process.  That floor is
# larger than most differences this study argues about.  If it can be removed,
# every sweep downstream becomes able to select; if it cannot, no amount of
# compute in stage 3 will settle anything and the design has to absorb the noise
# instead.
#
# The measurement is a replication, not an assertion: the SAME configuration is
# run three times in deterministic mode and twice in the published mode, and the
# out-of-fold vectors are compared bit-for-bit.  A claim of determinism that was
# only argued from the source would be exactly the kind of unverified mechanism
# this project has been caught by before.
#
#   MODE=det    sbatch --array=0-11 automl/slurm/determinism.sh   # 3 reps x 4 seeds
#   MODE=nondet sbatch --array=0-7  automl/slurm/determinism.sh   # 2 reps x 4 seeds
#
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
# Only read when the cuBLAS handle is created, so it has to be in the
# environment before python starts -- setting it inside the process is too late.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"

MODE="${MODE:?set MODE to det or nondet}"
# Four of the sixteen published seeds.  Fixed here rather than swept: this
# measures reproducibility of a configuration, so the seeds must be the same in
# every replicate or the comparison is between different models.
SEEDS=(7 11 23 37)
N=${#SEEDS[@]}
REP=$(( SLURM_ARRAY_TASK_ID / N ))
S=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N ))]}

case "${MODE}" in
  det)    FLAG="--deterministic" ;;
  nondet) FLAG="" ;;
  *)      echo "unknown MODE=${MODE}" >&2; exit 1 ;;
esac

OUT="${REPO}/automl/artifacts/determinism/${MODE}_rep${REP}"
mkdir -p "${OUT}"

# The published S0 configuration, unchanged: simplicial encoder, contrast
# objective, adjacent-pair checkpoint selection, 5 folds x 3 repeats.
echo "[determ] mode=${MODE} rep=${REP} seed=${S} flag='${FLAG}'"
python3 -m automl.topo.train --arch snn --tag "det_s${S}" \
    --pair-loss-weight 2.0 --select-on adjacent \
    --folds 5 --repeats 3 --seed "${S}" ${FLAG} --out-dir "${OUT}"
echo "DETERM DONE mode=${MODE} rep=${REP} seed=${S} $(date -Is)"
