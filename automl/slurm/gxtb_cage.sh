#!/bin/bash
#SBATCH --job-name=gxtb_cage
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtb_cage_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/gxtb_cage_%j.err
# g-xTB frozen-cage metal-swap probe, full modelled population. Resumable.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${REPO}"
# plumbing smoke first: two cages; abort the night if they fail
python3 -u -m automl.qc.gxtb_cage_probe --limit 2 --workers 2
N_OK=$(python3 -c "
import json, glob
ok = sum(1 for p in glob.glob('automl/artifacts/gxtb_cage/records/*.json')
         if json.load(open(p)).get('ok'))
print(ok)")
if [ "${N_OK}" -lt 1 ]; then echo 'SMOKE FAILED'; exit 1; fi
python3 -u -m automl.qc.gxtb_cage_probe --workers 48
echo "GXTB_CAGE DONE $(date -Is)"
