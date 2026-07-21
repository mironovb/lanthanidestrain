#!/bin/bash
#SBATCH --job-name=ln_qchg
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/qchg_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/qchg_%A_%a.err
#
# Mulliken charges for the re-optimised conformers -- see the module docstring.
#
# A single point at an existing geometry: no structure is changed, and single
# points converge where the ANCopt path did not.  This is what makes the ~3,100
# conformers usable as an ensemble instead of only as replacements.
#
#   SOLVENT=water   sbatch automl/slurm/conformer_charges.sh
#   SOLVENT=octanol sbatch automl/slurm/conformer_charges.sh
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
SOLVENT="${SOLVENT:-water}"
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

# xtb is single-threaded per structure here; the shard gets the parallelism.
python3 -m automl.qc.conformer_charges \
    --solvent "${SOLVENT}" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --threads "${SLURM_CPUS_PER_TASK}"

echo "QCHG DONE solvent=${SOLVENT} shard=${SLURM_ARRAY_TASK_ID} $(date -Is)"
