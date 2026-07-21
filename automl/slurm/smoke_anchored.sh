#!/bin/bash
#SBATCH --job-name=ln_smoke2
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:50:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/smoke2_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/smoke2_%j.err

set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONWARNINGS=ignore
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

python3 - <<'PY'
import time
from automl.matrix_cache import load_cache
from automl.experiment import ExperimentSpec, run_cv
from automl.evaluation import format_metrics

df, blocks, info = load_cache()
FAST = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
        "colsample_bytree": 0.5, "reg_lambda": 5.0}

trials = []
for preset in ("baseline_2d", "selectivity", "inner_sphere"):
    trials.append((f"flat        /{preset}",
                   ExperimentSpec(preset=preset, model="lgbm", params=FAST)))
    for level in ("extractant", "composition"):
        for sw in (1.0, 0.6):
            trials.append((
                f"anch-{level[:4]}-{sw}/{preset}",
                ExperimentSpec(preset=preset, model="anchored:lgbm",
                               params={**FAST, "level": level, "shape_weight": sw})))

for label, spec in trials:
    spec.repeats = 1
    spec.row_filter = "has3d"
    t = time.time()
    try:
        res = run_cv(df, blocks, spec, n_jobs=12)
        print(f"{label:34s} nfeat={int(res.metrics['n_features']):5d} "
              f"{format_metrics(res.metrics)}  [{time.time()-t:.0f}s]", flush=True)
    except Exception as exc:
        import traceback
        print(f"{label:34s} FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
PY
echo "SMOKE2 DONE $(date -Is)"
