#!/bin/bash
#SBATCH --job-name=ln_xtbchk
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/xtbchk_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/xtbchk_%j.err
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="$HOME/opt/xtb-dist/bin/xtb"
cd "${REPO}"
python3 -m automl.qc.xtb_backend --selftest --limit 3
echo "XTB SELFTEST DONE $(date -Is)"
