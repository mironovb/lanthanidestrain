#!/bin/bash
#SBATCH --job-name=inert_chk
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/inert_chk_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/inert_chk_%j.err
# Inertness proof for the --population edit: the exact published c15_plw4
# seed-201 config rerun under the edited train.py must reproduce the published
# OOF at max|delta| = 0.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
python3 -u -m automl.topo.train --arch dist --preset baseline_2d \
  --filtration-max 4.0 --heavy-only --pair-loss-weight 4.0 --rbf-bins 64 \
  --select-on adjacent --epochs 60 --folds 5 --repeats 3 --seed 201 \
  --deterministic --tag c15_plw4chk --out-dir automl/artifacts/topo_inert
python3 - <<'PY'
import pandas as pd, numpy as np, glob
a = pd.read_parquet(sorted(glob.glob('automl/artifacts/topo_c15/oof_c15_plw4_s201_*.parquet'))[0])
b = pd.read_parquet(sorted(glob.glob('automl/artifacts/topo_inert/oof_c15_plw4chk_*.parquet'))[0])
m = a.merge(b, on='safe_exp_id', suffixes=('_pub','_new'))
d = np.max(np.abs(m['oof_pub'] - m['oof_new']))
print(f"INERTNESS max|delta oof| = {d:.3e} over {len(m)} rows")
assert d == 0.0, "NOT INERT"
print("INERT: PASS")
PY
echo "INERT_CHK DONE $(date -Is)"
