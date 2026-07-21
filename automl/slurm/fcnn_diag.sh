#!/bin/bash
#SBATCH --job-name=ln_fdiag
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/fdiag_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/fdiag_%A_%a.err
# Post-hoc diagnostic on the published FCNN baseline -- see the module docstring.
# CPU partition, so it does not compete with the control factorial's GPU queue.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

MODES=(published grouped no_stop ensemble16 std_scaler)
M=${MODES[${SLURM_ARRAY_TASK_ID}]}
echo "[fdiag] mode=${M}"
python3 -m automl.topo.fcnn_diagnostic --modes "${M}"
echo "FDIAG DONE mode=${M} $(date -Is)"
