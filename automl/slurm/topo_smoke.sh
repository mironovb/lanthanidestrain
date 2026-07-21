#!/bin/bash
#SBATCH --job-name=ln_smoke
#SBATCH --partition=debug-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/toposmoke_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/toposmoke_%j.err
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Can the model fit at all?  If it cannot overfit 60 rows, nothing downstream matters.
python3 -m automl.topo.train --smoke --heavy-only --filtration-max 3.5
echo "SMOKE DONE $(date -Is)"
