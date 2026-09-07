#!/bin/bash
#SBATCH --job-name=lnxtb_cf
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/lnxtbcf_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/lnxtbcf_%A_%a.err
#
# Third Hamiltonian for the contraction benchmark: Ln-xTB (Zhang 2026,
# 10.1002/jcc.70321) -- stock GFN2 code with re-optimised La-Lu parameters
# (automl/qc/lnxtb_params.py).  Same 71 anchors, same 15-metal series, same
# protocol as cf_shard{0,1} (gfn2 / gxtb_hs).  Two arms: the SI's spin
# protocol (lnxtb_hs) and closed shell (lnxtb_cs) as the control.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_STACKSIZE=4G
cd "${REPO}"
python3 -u -m automl.qc.gxtb_series \
    --anchors "${ANCHORS:-200}" --max-atoms "${MAXATOMS:-300}" --workers 48 \
    --arms "${ARMS:-lnxtb_hs,lnxtb_cs}" --timeout "${TIMEOUT:-10800}" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NSHARDS:-2}" \
    --tag "${TAGBASE:-lnxtb_shard}${SLURM_ARRAY_TASK_ID:-0}"
echo "LNXTBCF DONE shard=${SLURM_ARRAY_TASK_ID:-0} rc=$? $(date -Is)"
