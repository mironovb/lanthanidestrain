#!/bin/bash
#SBATCH --job-name=ln_combo
#SBATCH --partition=debug-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:55:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/combo_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/combo_%j.err
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
out = Path("automl/artifacts/sweeps/combo")
presets = ["baseline_2d","plus_g14c","plus_g15c","cnfree_ligand",
           "cnfree_ligand_g13","g5_ligand","g5_ligand_g13",
           "core_ligand_cnfree","ligand3d_only"]
for preset in presets:
    for model, extra in (("lgbm", {}),
                         ("anchored:lgbm", {"shape_weight": 0.7}),
                         ("pairwise:lgbm", {"pair_key":"binned","delta_weight":0.6})):
        spec = ExperimentSpec(preset=preset, model=model, params={**P, **extra},
                              n_splits=5, repeats=3, seed=42,
                              row_filter="has3d", tag="combo")
        t=time.time()
        rec = run_and_record(df, blocks, spec, out, n_jobs=16, save_oof=True)
        if rec["status"]=="ok":
            m={k:v for k,v in rec["metrics"].items() if v is not None}
            print(f"{preset:20s} {model:14s} {format_metrics(m)} [{time.time()-t:.0f}s]", flush=True)
        else:
            print(f"{preset:20s} {model:14s} FAILED {rec.get('error')}", flush=True)
PY
echo "COMBO TEST DONE $(date -Is)"
