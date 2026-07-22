#!/bin/bash
#SBATCH --job-name=ln_pisw
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pisw_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/pisw_%A_%a.err
#
# Train ONE persistence-image configuration at ALL its seeds, on the TUNE half.
#
# Pre-registered in automl/reports/PI_SWEEP_PREREGISTRATION.md, committed before
# any run in this array existed.
#
# Every scope statement in this study declines to conclude anything from the
# PI-CNN's failure to replicate, because the images used the shipped asset's
# fixed settings and were never tuned. This gives them the tuning. The readout
# CNN, the objective (--pair-loss-weight 2.0 --select-on adjacent) and the folds
# are unchanged, so any difference is attributable to the representation.
#
# One task = one configuration = all seeds. A single run is only ~40 s, so a
# task-per-run would spend more time in the scheduler than on the GPU, and the
# account allows just 20 submitted tasks (MaxSubmitJobs=20). Bundling also means
# a configuration is never left with a partial seed set, which would ensemble
# over a different seed set than the configurations it is ranked against.
#
# AMENDED 22 July: runs train on ALL 162 extractants; only the SCORING is
# restricted to the tune half (in pi_sweep_test.py). --restrict-groups was the
# original design and it does not work: it removes 57% of the training rows
# (4,742 -> 2,030) and the PI-CNN collapses from adjR2 +0.1562 to +0.0362 on
# identical tune rows, so every configuration ties at stack weight 0.00 and the
# selection rule cannot rank anything. See PI_SWEEP_PREREGISTRATION.md #2a.
#
# PI_IMAGES_PATH selects the rendered image set; PersistenceImages already
# honours it, so pointing a run at a configuration needs no code change.
#
#   sbatch --array=0-24 automl/slurm/pi_sweep.sh <manifest.json> [seeds...]
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

MANIFEST=${1:?usage: pi_sweep.sh <manifest.json> [seed ...]}
shift || true
SEEDS=("$@")
# Eight seeds, drawn from the published matched set -- the same count the
# filtration replication used, so a sweep arm and a published arm are ensembled
# the same way.
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(7 11 23 37 42 51 67 83); fi

# Separate directory from the superseded tune-trained runs, which are kept as
# the evidence for the amendment rather than deleted.
OUT="${REPO}/automl/artifacts/pi_sweep/runs_full"
mkdir -p "${OUT}"

read -r KEY IMG < <(python3 -c "
import json
rows=json.load(open('${MANIFEST}'))
r=rows[${SLURM_ARRAY_TASK_ID}]
print(r['key'], r['path'])
")
export PI_IMAGES_PATH="${IMG}"
echo "[pisw] task=${SLURM_ARRAY_TASK_ID} key=${KEY} seeds=${SEEDS[*]}"
echo "[pisw] images=${IMG}"

rc=0
for S in "${SEEDS[@]}"; do
    # A single seed that fails must not abandon the rest: an incomplete cell is
    # reported as incomplete by the analysis (--min-seeds) rather than silently
    # averaged, so partial progress is still worth keeping.
    if python3 -m automl.topo.train --arch picnn --tag "sw${KEY}_s${S}" \
        --pair-loss-weight 2.0 --select-on adjacent \
        --folds 5 --repeats 3 --seed "${S}" --out-dir "${OUT}"; then
        echo "PISW SEED OK key=${KEY} seed=${S}"
    else
        echo "PISW SEED FAILED key=${KEY} seed=${S}"; rc=1
    fi
done
echo "PISW DONE key=${KEY} rc=${rc} $(date -Is)"
exit ${rc}
