#!/bin/bash
#SBATCH --job-name=ln_cnsub
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/cnsub_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/cnsub_%j.err

# Decisive test of the top recommendation.
#
# The stage-1 plan gives CN 9 to La-Gd and CN 8 to Tb-Lu, so every geometric
# descriptor carries a 5-17x discontinuity at Gd->Tb.  Inside EITHER subset
# that staircase does not exist.  If the 3D blocks help within a single-CN
# subset but not across the full series, the staircase is the blocker and
# removing it from the geometry generation would unlock the 3D information.
# If they still do not help, single-conformer noise dominates and conformer
# sampling is the priority instead.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONWARNINGS=ignore
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

python3 - <<'PY'
import time
from pathlib import Path
from automl.matrix_cache import load_cache
from automl.experiment import ExperimentSpec, run_and_record
from automl.evaluation import format_metrics
df, blocks, info = load_cache()
out = Path("automl/artifacts/sweeps/singlecn")
presets = ["baseline_2d", "plus_g1", "plus_g5", "core3d_qc", "inner_sphere",
           "plus_g14c", "all_3d"]
for rf in ("cn9_light", "cn8_heavy", "has3d"):
    for preset in presets:
        spec = ExperimentSpec(preset=preset, model="catboost", params={},
                              weight_scheme="group_inv", n_splits=5, repeats=3,
                              seed=42, row_filter=rf, tag="singlecn")
        t = time.time()
        rec = run_and_record(df, blocks, spec, out, n_jobs=16, save_oof=True)
        if rec["status"] == "ok":
            m = {k: v for k, v in rec["metrics"].items() if v is not None}
            print(f"{rf:10s} {preset:14s} {format_metrics(m)} [{time.time()-t:.0f}s]",
                  flush=True)
        else:
            print(f"{rf:10s} {preset:14s} FAILED {rec.get('error')}", flush=True)
PY
echo "SINGLECN TEST DONE $(date -Is)"
