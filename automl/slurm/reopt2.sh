#!/bin/bash
#SBATCH --job-name=ln_reopt2
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/reopt2_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/reopt2_%A_%a.err
# Stage 1, resharded for throughput.
#
# The first layout used 20 shards per solvent, ~62 structures each across 48
# workers.  Measured wall times are median 272 s but mean 565 s with a p90 of
# 1478 s, so a 62-job shard finishes its first wave and then drains -- ~34 of
# 48 cores sit idle waiting on the tail.  With only 2 nodes runnable at a time
# that projected to ~17 h.
#
# Four large shards (2 per solvent) instead: each task has ~618 structures for
# 48 workers, so the pool stays saturated and the idle tail is amortised over
# far more work.  Same total compute (~388 CPU-hours), ~4 h wall.
#
# Safe to run alongside/after the original array: reoptimize.py skips any
# structure that already has a completed status JSON, so the ~124 structures
# the first layout finished are not repeated.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="$HOME/opt/xtb-dist/bin/xtb"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${REPO}"

T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 2 ]; then SOLV=water;   SHARD=$T;
else                    SOLV=octanol; SHARD=$((T-2)); fi

echo "[reopt2] task=$T solvent=$SOLV shard=$SHARD/2 $(date -Is)"
python3 -m automl.qc.reoptimize \
    --solvent "$SOLV" --opt-level tight \
    --shard "$SHARD" --num-shards 2 \
    --threads 1 --workers 48 --timeout 5400
echo "REOPT2 DONE task=$T solvent=$SOLV $(date -Is)"
