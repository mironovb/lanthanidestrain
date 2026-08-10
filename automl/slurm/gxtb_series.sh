#!/bin/bash
#SBATCH --job-name=gxtb_ser
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtbser_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtbser_%j.err
#
# Does the OPTIMISED GEOMETRY carry f-shell structure, or only the wavefunction?
# Three arms, one binary, one protocol: GFN2 (f in core, uhf 0) / g-xTB high
# spin (f in valence, Hund uhf) / g-xTB closed shell (deliberate wrong-physics
# control -- if the break is f-shell, forcing uhf 0 should damage it).
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_STACKSIZE=4G
cd "${REPO}"
python3 -u -m automl.qc.gxtb_series \
    --anchors "${ANCHORS:-6}" --workers "${WORKERS:-48}" \
    --tag "${TAG:-opt_gas}" ${SOLVENT:+--solvent "${SOLVENT}"} ${EXTRA:-}
echo "GXTBSER DONE tag=${TAG:-opt_gas} rc=$? $(date -Is)"
