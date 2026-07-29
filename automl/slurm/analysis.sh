#!/bin/bash
# Generic runner for the read-only analysis modules.
#
#   sbatch automl/slurm/analysis.sh automl.topo.dualkey_test --n-boot 400
#
# These modules fit nothing and train nothing -- they read out-of-fold parquets
# and resample.  They are still submitted rather than run on the login node:
# a 400-draw cluster bootstrap over 162 extractants is minutes of CPU, and
# AGENTS.md reserves the login node for inspection and submission.
#SBATCH --job-name=ln_analysis
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:50:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/analysis_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/analysis_%j.err
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${REPO}"

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch automl/slurm/analysis.sh <module> [args...]" >&2
  exit 2
fi
MODULE="$1"; shift
echo "=== ${MODULE} $* ==="
echo "=== started $(date -Is) on $(hostname) ==="
# -u: unbuffered, so a long bootstrap streams its progress into the log instead
# of appearing all at once when the job exits.
python3 -u -m "${MODULE}" "$@"
RC=$?
echo "=== EXIT=${RC} finished $(date -Is) ==="
exit "${RC}"
