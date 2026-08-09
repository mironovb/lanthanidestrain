#!/bin/bash
#SBATCH --job-name=ln_c6nbr
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6nbr_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6nbr_%j.err
#
# Build the wider neighbour graphs.  CPU only, so it does not compete with the
# GPU confirmation runs for the single volta node the account may hold.
# Reads one file from read-only data/; writes only under automl/artifacts/.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -u -m automl.topo.build_neighbor_graph --all
echo "C6NBR DONE rc=$? $(date -Is)"
