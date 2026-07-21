#!/bin/bash
#SBATCH --job-name=ln_topotest
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/topotest_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/topotest_%j.err
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
cd "${REPO}"
python3 -m pytest automl/tests/test_simplicial.py -q --no-header -x 2>&1 | tail -30
echo "TOPO TESTS DONE $(date -Is)"
