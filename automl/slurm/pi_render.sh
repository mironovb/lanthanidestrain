#!/bin/bash
#SBATCH --job-name=ln_pirend
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pirend_%j.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pirend_%j.err
#
# Render one stage's persistence-image sets from the cached diagrams.
#
# Rendering itself is a matrix product (~2 s for all 953 complexes at 128x128);
# what actually costs time is compressing ~62 MB per configuration to disk. CPU
# only, and off the login node per AGENTS.md.
#
#   sbatch automl/slurm/pi_render.sh a
#   sbatch automl/slurm/pi_render.sh b <res> <spread>
#
# Stage A includes the shipped configuration as a reproduction anchor, so the
# later "tuned versus untuned" comparison is between two image sets built the
# same way in the same environment -- necessary because 18 of the 953 complexes
# are not bit-reproducible from the shipped asset here (see pi_sweep_build).
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

STAGE=${1:?usage: pi_render.sh <a|b> [res] [spread]}
if [ "${STAGE}" = "b" ]; then
    python3 -m automl.qc.pi_sweep_render --stage b --res "${2:?}" --spread "${3:?}"
else
    python3 -m automl.qc.pi_sweep_render --stage a
fi
echo "PIREND DONE stage=${STAGE} $(date -Is)"
