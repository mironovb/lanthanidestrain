#!/bin/bash
#SBATCH --job-name=gxtb_cf
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtbcf_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtbcf_%A_%a.err
#
# Per-ligand COMPLIANCE c_f = d(mean M-donor)/d(Shannon radius), across 71
# distinct ligands and the full 15-metal series, under GFN2 and under g-xTB.
#
# Why: on 6 ligands, 96.1% of g-xTB's non-linear metal response was shared
# across ligands -- a pure function of metal identity, which the model already
# has (metal identity is recoverable at R2=0.9995) and which therefore cannot
# be new information.  The only part that can be new is the LIGAND-SPECIFIC
# coefficient, whose spread was cv=0.059 on six ligands -- too few to tell
# physics from numerics.  This measures it on 71.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_STACKSIZE=4G
cd "${REPO}"
python3 -u -m automl.qc.gxtb_series \
    --anchors 200 --max-atoms "${MAXATOMS:-300}" --workers 48 \
    --arms "${ARMS:-gfn2,gxtb_hs}" --timeout "${TIMEOUT:-10800}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NSHARDS:-2}" \
    --tag "cf_shard${SLURM_ARRAY_TASK_ID}"
echo "GXTBCF DONE shard=${SLURM_ARRAY_TASK_ID} rc=$? $(date -Is)"
