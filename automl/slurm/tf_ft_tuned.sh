#!/bin/bash
#SBATCH --job-name=tf_fttuned
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_fttuned_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/tf_fttuned_%j.err
# The two best sweep settings at 4 seeds (legacy population, iteration set):
# small batches (bs 32, lr 5e-4, patience 20) alone and with the stronger
# regularisation (wd 1e-2, dropout 0.2).  POP=has3d for a confirmation run.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh; module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${REPO}"
POP="${POP:-ok_only}"
python3 -u -m automl.topo.anchored_ft --population "${POP}" --cells ft_anch --seeds ${SEEDS:-42 51 67 83} \
    --tag _bs32 --bs 32 --lr 5e-4 --patience 20 --epochs 300
python3 -u -m automl.topo.anchored_ft --population "${POP}" --cells ft_anch --seeds ${SEEDS:-42 51 67 83} \
    --tag _bs32wd --bs 32 --lr 5e-4 --patience 20 --epochs 300 --wd 1e-2 --dropout 0.2
echo "FT_TUNED DONE pop=${POP} $(date -Is)"
