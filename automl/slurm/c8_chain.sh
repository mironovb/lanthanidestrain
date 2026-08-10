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

# Hops-kept variant (gxtbh/shiph).  Dropping basin hops makes the contrast
# cleaner but costs ~17% of the complexes, and the preliminary run showed BOTH
# arms collapse to a negative R2 at 406 complexes -- so complex count is not a
# free parameter and the trade-off is measured rather than assumed.
python3 -u - <<'PYEOF'
import sys; sys.argv=['x']
from automl.topo.build_vr_gxtb import eligible, build, OUT_ROOT
import shutil, numpy as np
keep = eligible(drop_hops=False)
for arm, src in (("gxtbh","gxtb"), ("shiph","ship")):
    build(src, 4.0, False, keep=list(keep), verbose=False)
    d = OUT_ROOT/arm; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT_ROOT/src/"vietoris_rips_inputs.npz", d/"vietoris_rips_inputs.npz")
    shutil.copy(OUT_ROOT/src/"meta.json", d/"meta.json")
a=np.load(OUT_ROOT/"gxtbh/vietoris_rips_inputs.npz"); b=np.load(OUT_ROOT/"shiph/vietoris_rips_inputs.npz")
assert np.array_equal(a["build_ids"], b["build_ids"]), "hops-kept arms not matched"
print(f"[c8chain] hops-kept: {len(a['build_ids'])} complexes, arms matched")
PYEOF
# The unsuffixed pair must be rebuilt last: the loop above overwrote them.
python3 -u -m automl.topo.build_vr_gxtb --both --cutoff 4.0 || exit 1

# Pre-build every triangle-edge cache SERIALLY.  SimplicialComplexes writes
# triangle_edges.tmp.npz and atomically renames it; two cells starting on the
# same asset at once race, one wins the rename and the other dies with
# FileNotFoundError on its own temp file.  That already cost a cell in the
# snapshot run.  Building them here, one at a time, means every campaign cell
# finds a warm cache and never creates one.
for ARM in gxtb ship gxtbh shiph; do
  python3 - "${ARM}" <<'PYEOF'
import sys
from automl.topo.train import edge_asset
a = edge_asset(sys.argv[1])
print(f"[c8chain] triangle cache warm for {sys.argv[1]}")
PYEOF
done

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
