#!/bin/bash
#SBATCH --job-name=ln_smoke
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/smoke_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/smoke_%j.err

set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONWARNINGS=ignore
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

python3 - <<'PY'
import time
import numpy as np
from automl.matrix_cache import load_cache
from automl.experiment import ExperimentSpec, run_cv
from automl.evaluation import format_metrics

df, blocks, info = load_cache()
FAST = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
        "colsample_bytree": 0.5, "reg_lambda": 5.0}

trials = [
    ("flat      / baseline_2d", ExperimentSpec(preset="baseline_2d", model="lgbm", params=FAST)),
    ("flat      / selectivity", ExperimentSpec(preset="selectivity", model="lgbm", params=FAST)),
    ("twostage  / baseline_2d", ExperimentSpec(preset="baseline_2d", model="twostage:lgbm", params=FAST)),
    ("twostage  / selectivity", ExperimentSpec(preset="selectivity", model="twostage:lgbm", params=FAST)),
    ("pairwise  / baseline_2d", ExperimentSpec(preset="baseline_2d", model="pairwise:lgbm", params=FAST)),
    ("pairwise  / inner_sphere", ExperimentSpec(preset="inner_sphere", model="pairwise:lgbm", params=FAST)),
    ("pairwise  / selectivity", ExperimentSpec(preset="selectivity", model="pairwise:lgbm", params=FAST)),
    ("pairwise05/ selectivity", ExperimentSpec(preset="selectivity", model="pairwise:lgbm",
                                               params={**FAST, "delta_weight": 0.5})),
]
for label, spec in trials:
    spec.repeats = 1
    spec.row_filter = "has3d"
    t = time.time()
    try:
        res = run_cv(df, blocks, spec, n_jobs=12)
        print(f"{label:26s} nfeat={int(res.metrics['n_features']):5d} "
              f"{format_metrics(res.metrics)}  [{time.time()-t:.0f}s]", flush=True)
    except Exception as exc:
        import traceback
        print(f"{label:26s} FAILED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
PY
echo "SMOKE DONE $(date -Is)"
