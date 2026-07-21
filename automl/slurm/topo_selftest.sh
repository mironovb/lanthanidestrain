#!/bin/bash
#SBATCH --job-name=ln_topoself
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/topoself_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/topoself_%j.err
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
cd "${REPO}"
python3 -m automl.topo.simplicial_data --selftest
echo "--- sparsification stats ---"
python3 -m automl.topo.simplicial_data --stats
echo "TOPO SELFTEST DONE $(date -Is)"
