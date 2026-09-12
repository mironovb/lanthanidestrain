#!/bin/bash
#SBATCH --job-name=tf_tab
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_tab_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_tab_%j.err
# Tabular transformer (FT-Transformer level/shape) and block transformer
# (shape only), leave-extractants-out 5x3, seed ensembles.
#   POP  ok_only (default) | has3d
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
POP="${POP:-ok_only}"
python3 -u -m automl.topo.block_transformer --population "${POP}" \
    --seeds ${BT_SEEDS:-42 51 67 83 91 103 107 109}
python3 -u -m automl.topo.anchored_ft --population "${POP}" \
    --cells ft_anch ft_flat --seeds ${FT_SEEDS:-42 51 67 83}
echo "TF_TAB DONE pop=${POP} $(date -Is)"
