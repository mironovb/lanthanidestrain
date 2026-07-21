#!/bin/bash
#SBATCH --job-name=ln_s2
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/s2_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/s2_%A_%a.err
#
# S2: the variance-reduced simplicial arm.  Pre-registered in
# automl/reports/S2_PREREGISTRATION.md, committed before this was submitted.
#
# The SNN is the noisiest arm in the control factorial (per-seed SD 0.047 vs the
# tabular control's 0.027) with 1.11M parameters on 953 distinct geometries.
# Four levers, all aimed at that:
#
#   --conformers 3     2,797 structures over 956 complexes; a random conformer
#                      per complex per epoch, all of them mean-pooled at
#                      inference
#   --block-centre     concatenate each embedding with its deviation from the
#                      composition-block mean, cancelling common-mode ligand and
#                      conformer noise
#   --pretrain-epochs  masked-charge / edge-radius reconstruction over every
#                      conformer, no log D involved
#   32 seeds           the SNN gains most of any arm from averaging
#
# Memory is raised to 96G: three VR assets are resident at once (~55 + 52 + 46 MB
# of npz plus their boundary maps and the per-conformer complex cache).
#
#   sbatch --array=0-31 automl/slurm/topo_s2.sh
#
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_s2"
mkdir -p "${OUT}"

# The first 16 are the control factorial's matched set, so S2 - S0 stays
# seed-paired; the second 16 extend the ensemble.
SEEDS=(7 11 23 37 42 51 67 83 211 223 233 241 251 263 271 281
       307 311 313 317 331 337 347 349 353 359 367 373 379 383 389 397)
S=${SEEDS[${SLURM_ARRAY_TASK_ID}]}

echo "[s2] seed=${S} conformers=3 block-centre pretrain=${PRETRAIN:-20}"
python3 -m automl.topo.train \
    --arch snn --tag "s2_s${S}" \
    --pair-loss-weight 2.0 --select-on adjacent \
    --conformers 3 --block-centre \
    --pretrain-epochs "${PRETRAIN:-20}" \
    --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"
echo "S2 DONE seed=${S} $(date -Is)"
