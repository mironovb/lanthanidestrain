#!/bin/bash
#SBATCH --job-name=c18_3dtab
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=11:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c18_%A_%a.out
# The strengthened-2D control: the same winning CatBoost config, PLUS the
# 3D-derived tabular blocks (qc + polyhedron + complex-physical). Separates
# "the VR encoder extracts something cheap descriptors cannot" from "any 3D
# information would have done" -- a distinction the current stack comparison
# conflates, because baseline_2d carries no 3D-derived columns at all.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"
PRESETS=(plus_p3d_all plus_p3d_phys)
P="${PRESETS[${SLURM_ARRAY_TASK_ID}]}"
python3 -u -m automl.topo.c6_partners --which catboost --only q60_rsm03_deep \
    --preset "${P}" --seeds 8 --repeats 3 --restrict full
echo "C18 DONE ${P} rc=$? $(date -Is)"
