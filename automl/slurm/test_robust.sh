#!/bin/bash
#SBATCH --job-name=ln_robust
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/robust_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/robust_%j.err

# Robust-target test.  0.83 % of rows sit below log D = -4 and carry 9.1 % of the
# total variance; 47 of those 50 rows belong to a single extractant (C5BTBP),
# which a leave-extractants-out fold can never see.  Test whether winsorising
# the training target, or a robust objective, buys held-out accuracy.
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
P = {"n_estimators": 900, "learning_rate": 0.04, "num_leaves": 63,
     "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.5,
     "reg_lambda": 5.0}
out = Path("automl/artifacts/sweeps/robust")

trials = []
for preset in ("baseline_2d", "plus_g5", "plus_g15c"):
    for model, extra in (("lgbm", {}), ("anchored:lgbm", {"shape_weight": 0.7})):
        for clip in (0.0, 6.0, 4.0, 3.0):
            trials.append((preset, model, {**P, **extra}, clip, "none"))
    # robust objectives at no clipping
    for obj in ("huber", "fair"):
        trials.append((preset, "lgbm", {**P, "objective": obj}, 0.0, "none"))
    # sample-weight schemes on the strongest architecture
    for w in ("group_inv", "target_lds", "combo"):
        trials.append((preset, "anchored:lgbm", {**P, "shape_weight": 0.7}, 0.0, w))

for preset, model, params, clip, wsch in trials:
    spec = ExperimentSpec(preset=preset, model=model, params=params,
                          weight_scheme=wsch, n_splits=5, repeats=3, seed=42,
                          row_filter="has3d", target_clip=clip, tag="robust")
    t = time.time()
    rec = run_and_record(df, blocks, spec, out, n_jobs=16, save_oof=True)
    obj = params.get("objective", "l2")
    tagline = f"{preset:12s} {model:14s} clip={clip:<4} obj={obj:<6} w={wsch:10s}"
    if rec["status"] == "ok":
        m = {k: v for k, v in rec["metrics"].items() if v is not None}
        print(f"{tagline} {format_metrics(m)} [{time.time()-t:.0f}s]", flush=True)
    else:
        print(f"{tagline} FAILED {rec.get('error')}", flush=True)
PY
echo "ROBUST TEST DONE $(date -Is)"
