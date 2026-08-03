#!/bin/bash
#SBATCH --job-name=ln_neut
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/neut_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/neut_%A_%a.err
#
# Pre-registered in automl/reports/CAMPAIGN4_PREREGISTRATION.md.
#
#   MODE=pilot sbatch automl/slurm/neutralize.sh            # RUN THIS FIRST
#   sbatch --array=0-7 automl/slurm/neutralize.sh
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export XTB_BIN="$HOME/opt/xtb-dist/bin/xtb"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${REPO}"

if [[ "${MODE:-run}" == "smoke" ]]; then
  echo "=== smoke: 2 structures, both arms, full ladder ==="
  python3 -u -m automl.qc.neutralize --pilot 2 --workers 2 --timeout 3600 --overwrite
  echo "SMOKE DONE $(date -Is)"
  exit 0
fi
if [[ "${MODE:-run}" == "charges" ]]; then
  # second pass: one GFN2 single point per arm, to attach per-atom Mulliken
  # charges.  Kept separate so the optimisations are never repeated for it.
  python3 -u -m automl.qc.neutralize --charges \
      --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NSHARDS:-1}" \
      --workers "${SLURM_CPUS_PER_TASK}"
  echo "CHARGES DONE $(date -Is)"
  exit 0
fi
if [[ "${MODE:-run}" == "pilot" ]]; then
  python3 -u -m automl.qc.neutralize --pilot "${PILOT:-40}" \
      --workers "${SLURM_CPUS_PER_TASK}" --timeout 14400
  echo "PILOT DONE $(date -Is)"
  exit 0
fi
python3 -u -m automl.qc.neutralize --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "${NSHARDS:-8}" --workers "${SLURM_CPUS_PER_TASK}" --timeout 14400
echo "NEUT DONE shard=${SLURM_ARRAY_TASK_ID} $(date -Is)"
