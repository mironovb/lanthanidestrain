#!/bin/bash
#SBATCH --job-name=pair_sweep
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pair_sweep_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pair_sweep_%j.err
#
# Phase B sweep: the all-pairs delta model over loss x adjacent-weight, plus
# the strict-key and expanded-population variants.  CPU only.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

python3 -u -m automl.topo.pair_model --sweep --repeats 3
# expanded population (has3d) with the winning-so-far config family
python3 -u -m automl.topo.pair_model --population has3d --loss mae --adj-weight 3 --repeats 3
python3 -u -m automl.topo.pair_model --population has3d --loss q0.6 --adj-weight 3 --repeats 3
echo "PAIR_SWEEP DONE $(date -Is)"
