#!/bin/bash
#SBATCH --job-name=ln_cnfree
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/cnfree_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/cnfree_%j.err

# Focused test of the coordination-number-artefact hypothesis:
# if the CN staircase imposed by cn_for_Z is what breaks the series ordering,
# then removing the CN main effect (block g15) should restore the pairwise
# separation-factor R^2 that every raw 3D block destroys.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONWARNINGS=ignore
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

python3 - <<'PY'
import time
from automl.matrix_cache import load_cache
from automl.experiment import ExperimentSpec, run_and_record
from automl.evaluation import format_metrics
from pathlib import Path

df, blocks, info = load_cache()
P = {"n_estimators": 900, "learning_rate": 0.04, "num_leaves": 63,
     "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.5,
     "reg_lambda": 5.0}
out = Path("automl/artifacts/sweeps/cnfree")

presets = ["baseline_2d",          # reference
           "plus_g1",              # raw block that carries the CN staircase
           "plus_g5",              # the block that does NOT carry it
           "plus_g15c", "plus_g15",# CN main effect regressed out
           "cnfree", "cnfree_full",
           "core3d_qc", "core3d_cnfree", "core3d_cnfree_lig",
           "plus_g12c", "denoised", "ligand3d_only"]

for preset in presets:
    for model, extra in (("lgbm", {}),
                         ("anchored:lgbm", {"shape_weight": 0.7})):
        spec = ExperimentSpec(preset=preset, model=model, params={**P, **extra},
                              n_splits=5, repeats=3, seed=42,
                              row_filter="has3d", tag="cnfree")
        t = time.time()
        rec = run_and_record(df, blocks, spec, out, n_jobs=16, save_oof=True)
        if rec["status"] == "ok":
            m = {k: v for k, v in rec["metrics"].items() if v is not None}
            print(f"{preset:20s} {model:14s} {format_metrics(m)} [{time.time()-t:.0f}s]",
                  flush=True)
        else:
            print(f"{preset:20s} {model:14s} FAILED {rec.get('error')}", flush=True)
PY
echo "CNFREE TEST DONE $(date -Is)"
