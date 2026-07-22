#!/bin/bash
#SBATCH --job-name=ln_filt
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/filt_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/filt_%A_%a.err
#
# A THIRD topological view, to bound what the positive result is about.
#
# SYNTHESIS.md names this the single most informative remaining experiment. The
# stack result is currently specific to one encoder: the simplicial network at
# filtration 3.5 A. The persistence-image CNN failed to replicate it, which
# leaves two live readings:
#
#   (a) the finding is about MESSAGE PASSING OVER A VIETORIS-RIPS COMPLEX, in
#       which case a different filtration radius should also add; or
#   (b) it is about one specific complex construction, in which case it should
#       not, and the claim narrows again.
#
# Same architecture and objective as S0; only --filtration-max changes, so the
# comparison isolates the complex rather than the model. Two radii either side
# of the S0 default (3.5): a tighter 3.0 (coordination sphere only) and a looser
# 4.0 (the asset's own max edge). 8 seeds each, drawn from the published S0
# matched set so every contrast stays seed-paired.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_filt"
mkdir -p "${OUT}"

SEEDS=(7 11 23 37 42 51 67 83)
T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 8 ]; then F=3.0; S=${SEEDS[$T]}; else F=4.0; S=${SEEDS[$((T-8))]}; fi

echo "[filt] filtration=${F} seed=${S}"
python3 -m automl.topo.train --arch snn --tag "filt${F}_s${S}" \
    --pair-loss-weight 2.0 --select-on adjacent \
    --filtration-max "${F}" \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "FILT DONE f=${F} seed=${S} $(date -Is)"
