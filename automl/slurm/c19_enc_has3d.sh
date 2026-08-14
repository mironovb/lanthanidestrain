#!/bin/bash
#SBATCH --job-name=c19_enc
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c19_enc_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c19_enc_%j.err
#
# C19: the c15_plw4 distance encoder on the EXPANDED population (has3d rows,
# has3d edge asset).  Its OOF covers the fresh-444 rows, enabling the I15
# anchored-3D blend to be scored on the frozen confirmation pairs.
# Submit ONLY after inert_check passes (max|delta oof| = 0).
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
OUT="${REPO}/automl/artifacts/topo_c19"
mkdir -p "${OUT}"
for SEED in 42 51 67 83 91 103 107 109; do
  python3 -u -m automl.topo.train --arch dist --preset baseline_2d \
    --filtration-max 4.0 --heavy-only --pair-loss-weight 4.0 --rbf-bins 64 \
    --select-on adjacent --epochs 60 --folds 5 --repeats 3 --seed "${SEED}" \
    --deterministic --edge-asset has3d --population has3d \
    --tag c19_plw4h3d --out-dir "${OUT}"
done
echo "C19_ENC DONE $(date -Is)"
