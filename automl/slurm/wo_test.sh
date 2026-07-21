#!/bin/bash
#SBATCH --job-name=ln_wo
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/wo_%A.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/wo_%A.err
# The pre-registered water<->octanol test: 4 CatBoost arms x 5x3 folds, paired
# cluster bootstrap over extractants. CPU only.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
python3 -m automl.topo.water_octanol_test --n-boot 400 --n-jobs "${SLURM_CPUS_PER_TASK}"
echo "WO DONE $(date -Is)"
