#!/bin/bash
#SBATCH --job-name=ln_qcctl
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/qcctl_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/qcctl_%j.err

# CONFOUND CONTROL for the "3D helps Tb-Lu" result.
#
# Every plus_* preset bundles the `qc` block (geometry QC class one-hot).  The
# two halves of the series have very different QC profiles:
#   La-Gd : 87.6 % OK, 7.9 % BORDERLINE_LONGISH, 0.7 % AMBIGUOUS_SHELL
#   Tb-Lu : 64.0 % OK, 1.0 % BORDERLINE_LONGISH, 33.0 % AMBIGUOUS_SHELL
# so the QC flag is near-constant in the light half and highly informative in
# the heavy half.  If `baseline_2d_qc` (2D + qc, NO 3D descriptors) captures the
# heavy-half gain on its own, then "3D helps the heavy lanthanides" is really
# "the geometry QC flag is informative for the heavy lanthanides".
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONWARNINGS=ignore
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

python3 - <<'PY'
import time
from pathlib import Path
from automl.matrix_cache import load_cache
from automl.experiment import ExperimentSpec, run_and_record
from automl.evaluation import format_metrics
df, blocks, info = load_cache()
out = Path("automl/artifacts/sweeps/qcctl")
# baseline_2d      : no qc, no 3D
# baseline_2d_qc   : qc only, no 3D   <- the control that matters
# plus_g5 / core3d_qc : qc + 3D
presets = ["baseline_2d", "baseline_2d_qc", "plus_g5", "core3d_qc"]
for rf in ("cn8_heavy", "cn9_light"):
    for preset in presets:
        spec = ExperimentSpec(preset=preset, model="catboost", params={},
                              weight_scheme="group_inv", n_splits=5, repeats=3,
                              seed=42, row_filter=rf, tag="qcctl")
        t = time.time()
        rec = run_and_record(df, blocks, spec, out, n_jobs=8, save_oof=True)
        if rec["status"] == "ok":
            m = {k: v for k, v in rec["metrics"].items() if v is not None}
            print(f"{rf:10s} {preset:16s} {format_metrics(m)} [{time.time()-t:.0f}s]",
                  flush=True)
        else:
            print(f"{rf:10s} {preset:16s} FAILED {rec.get('error')}", flush=True)
PY
echo "QC CONTROL DONE $(date -Is)"
