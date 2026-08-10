#!/bin/bash
#SBATCH --job-name=c8_chain
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c8chain_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c8chain_%j.err
#
# Rebuild the gxtb/ship assets from ALL finished re-optimisations, verify the
# two arms are matched, and only then release the training campaign.
#
# The gate is the point.  If the arms differ in complex set, order, atom count
# or composition, the contrast compares datasets rather than geometries -- the
# single way this experiment could return a confident wrong answer.  That
# already happened once (398 vs 399 complexes, because the re-optimisation was
# still writing records between the two build() calls).  So the campaign is
# NOT submitted unless --verify passes.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

echo "[c8chain] re-opt records: $(ls automl/artifacts/gxtb_reopt/records/*.json 2>/dev/null | wc -l)"
python3 -u -m automl.topo.build_vr_gxtb --both --cutoff 4.0
rc=$?
if [[ ${rc} -ne 0 ]]; then
  echo "[c8chain] ASSETS NOT MATCHED (rc=${rc}) -- campaign NOT submitted"
  exit "${rc}"
fi
python3 -u -m automl.topo.build_vr_gxtb --basin-report

OUT="${REPO}/automl/artifacts/topo_c8"
mkdir -p "${OUT}"
# Preliminary cells were run on a 406-complex subset; their sentinels must not
# suppress the full-asset reruns, and their rows must not be pooled with them.
rm -f "${OUT}"/.done/c8pre_* 2>/dev/null || true

MANIFEST="${REPO}/automl/slurm/manifests/c8_night.json"
N=$(python3 -c "import json;print(len(json.load(open('${MANIFEST}'))))")
K="${CELLS_PER_TASK:-4}"
TASKS=$(( (N + K - 1) / K ))
echo "[c8chain] ${N} cells / ${K} per task = ${TASKS} array tasks"

# Both GPU partitions have their OWN GrpTRES and MaxSubmit accounting, so a
# driver on each runs concurrently without competing.
A=$(sbatch --parsable --array=0-$(( TASKS/2 )) \
    --export=ALL,MANIFEST="${MANIFEST}",CELLS_PER_TASK="${K}",OUT_DIR="${OUT}" \
    "${REPO}/automl/slurm/c6_cells.sh")
B=$(sbatch --parsable --array=$(( TASKS/2 + 1 ))-$(( TASKS - 1 )) \
    --export=ALL,MANIFEST="${MANIFEST}",CELLS_PER_TASK="${K}",OUT_DIR="${OUT}" \
    "${REPO}/automl/slurm/c6_cells_debug.sh")
echo "[c8chain] submitted volta=${A} debug-gpu=${B}"
echo "C8CHAIN DONE $(date -Is)"
