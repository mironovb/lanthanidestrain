#!/bin/bash
#SBATCH --job-name=ln_reoptp
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:55:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/reoptpilot_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/reoptpilot_%j.err
# Timing pilot: how long does a properly converged, solvated optimisation take?
# Runs 12 structures across the size range before committing the full array.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="$HOME/opt/xtb-dist/bin/xtb"
cd "${REPO}"
python3 -m automl.qc.reoptimize --solvent water --opt-level tight \
    --shard 0 --num-shards 100 --limit 1200 --threads 1 --timeout 5400 \
    --workers 12
echo "REOPT PILOT DONE $(date -Is)"
