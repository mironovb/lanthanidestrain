#!/bin/bash
#SBATCH --job-name=ln_match
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/match_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/match_%j.err

# Does 3D help the heavy lanthanides because of chemistry, or because the heavy
# half has half the ligand coverage and a data-starved 2D model?
#
# Subsample La-Gd to the Tb-Lu coverage profile (same 82 extractants, same 20
# with >=10 rows, matched size distribution) and re-run.  Three independent
# matched draws.  If the 3D benefit appears in the light half at matched sample
# size, it is data thinness.  If it stays negative, it is chemistry.
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
out = Path("automl/artifacts/sweeps/matched")
presets = ["baseline_2d", "plus_g1", "plus_g5", "core3d_qc", "inner_sphere"]
# three matched draws of the light half, plus the heavy half as the control
for rf in ("cn9_matched:0", "cn9_matched:1", "cn9_matched:2", "cn8_heavy"):
    for preset in presets:
        spec = ExperimentSpec(preset=preset, model="catboost", params={},
                              weight_scheme="group_inv", n_splits=5, repeats=3,
                              seed=42, row_filter=rf, tag="matched")
        t = time.time()
        rec = run_and_record(df, blocks, spec, out, n_jobs=16, save_oof=True)
        if rec["status"] == "ok":
            m = {k: v for k, v in rec["metrics"].items() if v is not None}
            print(f"{rf:15s} {preset:14s} {format_metrics(m)} [{time.time()-t:.0f}s]",
                  flush=True)
        else:
            print(f"{rf:15s} {preset:14s} FAILED {rec.get('error')}", flush=True)
PY
echo "MATCHED TEST DONE $(date -Is)"
