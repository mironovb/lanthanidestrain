#!/bin/bash
#SBATCH --job-name=ln_tests
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:50:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tests_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tests_%j.err
#
# The suite takes 5+ minutes of CPU and AGENTS.md keeps heavy work off the login
# node.  Running it here also stops the output being truncated when the login
# node is busy, which is how a green run came back looking like a failure.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="$HOME/opt/xtb-dist/bin/xtb"
cd "${REPO}"
python3 -m pytest automl/tests -q --no-header
echo "TESTS EXIT=$? $(date -Is)"
