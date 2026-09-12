#!/bin/bash
#SBATCH --job-name=tf_ftsweep
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_ftsweep_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_ftsweep_%j.err
# FT-Transformer tuning sweep on the legacy population (iteration set), one
# seed per variant, level/shape cell.  Reference: default s42 +0.078 / 0.269.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh; module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
run() { tag=$1; shift; echo "=== ${tag}: $* ==="; python3 -u -m automl.topo.anchored_ft --cells ft_anch --seeds 42 --tag "${tag}" "$@"; }
run _p20            --patience 20 --epochs 300
run _lr3e-4_p20     --lr 3e-4 --patience 20 --epochs 300
run _d128_l4_p20    --dim 128 --layers 4 --dropout 0.2 --patience 20 --epochs 300
run _bt32_p20       --bit-tokens 32 --patience 20 --epochs 300
run _wd1e-2_p20     --wd 1e-2 --dropout 0.2 --patience 20 --epochs 300
run _bs32_p20       --bs 32 --lr 5e-4 --patience 20 --epochs 300
echo "FT_SWEEP DONE $(date -Is)"
