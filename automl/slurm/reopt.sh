#!/bin/bash
#SBATCH --job-name=ln_reopt
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --array=0-39
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/reopt_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/reopt_%A_%a.err
# Stage 1: re-optimise every complex at tight convergence in implicit solvent.
#
# The shipped geometries all stopped on a loose fmax = 0.2 eV/A criterion.  The
# timing pilot re-optimised 7 of them to fmax = 0.0009-0.009 eV/A (20-200x
# tighter), every one meeting target, at a median ~100 s per structure.
#
# Two solvents, because log D is a partition coefficient and a single phase is
# the wrong reference state for it: tasks 0-19 do water, 20-39 do n-octanol.
# The water-minus-octanol difference then becomes a descriptor block in its own
# right rather than merely a tidier geometry.
#
# xtb runs single-threaded, so throughput comes from running 48 structures at
# once per node rather than from parallelising any one of them.  Fully
# resumable: a structure with a completed status JSON is skipped, so a requeued
# or extended task picks up exactly where it stopped.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="$HOME/opt/xtb-dist/bin/xtb"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${REPO}"

T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 20 ]; then SOLV=water;   SHARD=$T;
else                     SOLV=octanol; SHARD=$((T-20)); fi

echo "[reopt] task=$T solvent=$SOLV shard=$SHARD/20 $(date -Is)"
python3 -m automl.qc.reoptimize \
    --solvent "$SOLV" --opt-level tight \
    --shard "$SHARD" --num-shards 20 \
    --threads 1 --workers 48 --timeout 5400
echo "REOPT DONE task=$T solvent=$SOLV $(date -Is)"
