#!/bin/bash
#SBATCH --job-name=ln_pifin
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pifin_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pifin_%A_%a.err
#
# Stage C: the single winning configuration, 16 seeds, ALL 162 extractants.
#
# This is the only run the confirm half ever sees, and it is deliberately the
# last thing that happens. Note what is NOT passed here: --restrict-groups. The
# sweep was confined to the tune half so that selection never touched the
# confirm extractants; now that the configuration is chosen and frozen, the
# winner is trained on everything, exactly as the published arms were.
#
# That ordering is the whole design. Selection on disjoint extractants is what
# lets the confirmatory interval skip a multiplicity penalty for the ~49
# configurations swept -- so this script must never run before a winner has been
# fixed from the tune half alone.
#
# 16 seeds and the same matched set the published S0 and T0w arms use, so the
# arm being tested and the arms it is compared against are ensembled alike.
#
#   sbatch --array=0-15 automl/slurm/pi_final.sh <images.npz>
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

IMG=${1:?usage: pi_final.sh <path to img_<key>.npz>}
[ -f "${IMG}" ] || { echo "no such image set: ${IMG}"; exit 2; }
KEY=$(basename "${IMG}" .npz); KEY=${KEY#img_}

SEEDS=(7 11 23 37 42 51 67 83 211 223 233 241 251 263 271 281)
S=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
OUT="${REPO}/automl/artifacts/pi_final"
mkdir -p "${OUT}"

export PI_IMAGES_PATH="${IMG}"
echo "[pifin] winner=${KEY} seed=${S} images=${IMG}"
echo "[pifin] FULL data -- no --restrict-groups, by design"

python3 -m automl.topo.train --arch picnn --tag "fin${KEY}_s${S}" \
    --pair-loss-weight 2.0 --select-on adjacent \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"

echo "PIFIN DONE key=${KEY} seed=${S} $(date -Is)"
