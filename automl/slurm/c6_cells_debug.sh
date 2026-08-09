#!/bin/bash
#SBATCH --job-name=ln_c6d
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6d_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/c6d_%A_%a.err
#
# CAMPAIGN6 cell runner, debug-gpu variant.  Identical to c6_cells.sh except
# for the partition and the 2 h wall clock.
#
# Why it exists: debug-gpu has its OWN GrpTRES node=1 and its own MaxSubmit
# accounting, and campaign_driver.sh's `mine` filters squeue by partition.  So
# a driver on this script and a driver on c6_cells.sh run concurrently without
# competing, roughly doubling GPU throughput.
#
# The 2 h limit is a HARD kill, so CELLS_PER_TASK must be sized against the
# SLOWEST cell in the slice, not the mean.  Send screening cells
# (--repeats 1, --arch dist, ~3 min) here; keep --repeats 3 cells on volta.
#
# CAMPAIGN6 cell runner: K cells per array task, run SEQUENTIALLY.
#
# Why pack cells.  MaxSubmitJobs=40 counts ARRAY TASKS, not jobs (see
# campaign_driver.sh's `mine`), so a 300-cell campaign cannot be one array
# however it is chunked.  K cells per task divides the task count by K and
# leaves the driver's drip-feed doing what it already does well.
#
#   automl/slurm/campaign_driver.sh automl/slurm/c6_cells.sh <ntasks> 8 34 \
#       MANIFEST=automl/slurm/manifests/c6_w1.json CELLS_PER_TASK=6
#
# NOT `set -e`: one failing cell must not take the other K-1 with it.  Failures
# are printed and returned in the exit status; they are not hidden.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
cd "${REPO}"

MANIFEST="${MANIFEST:?set MANIFEST=<path to cells json>}"
K="${CELLS_PER_TASK:-3}"
# NEVER the default --out-dir.  train.py APPENDS to <out-dir>/results.jsonl and
# automl/artifacts/topo_runs/results.jsonl is SHA-pinned by control_guard; one
# run with the default out-dir fails --verify unrecoverably.
OUT="${OUT_DIR:-${REPO}/automl/artifacts/topo_c6}"
mkdir -p "${OUT}" "${OUT}/.done"

LO=$(( SLURM_ARRAY_TASK_ID * K ))
HI=$(( LO + K - 1 ))
echo "[c6] task ${SLURM_ARRAY_TASK_ID} -> cells ${LO}..${HI} of ${MANIFEST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

rc_all=0
for (( i=LO; i<=HI; i++ )); do
  LINE=$(python3 -c '
import json,sys
cells=json.load(open(sys.argv[1]))
i=int(sys.argv[2])
if i>=len(cells): raise SystemExit(0)
c=cells[i]; print(c["tag"]+"\t"+c["args"])' "${MANIFEST}" "${i}")
  if [[ -z "${LINE}" ]]; then echo "[c6] cell ${i} past end of manifest"; break; fi
  TAG="${LINE%%$'\t'*}"; ARGS="${LINE#*$'\t'}"
  # Resumability.  The run STEM is built inside train.py from a dozen flags and
  # is not knowable here, so the sentinel is keyed on the tag, which the
  # manifest generator guarantees is unique per cell.
  if [[ -f "${OUT}/.done/${TAG}" ]]; then
    echo "[c6] cell ${i} ${TAG}: already done, skipping"; continue
  fi
  echo "[c6] cell ${i} ${TAG}: ${ARGS}"
  t0=$SECONDS
  python3 -u -m automl.topo.train ${ARGS} --tag "${TAG}" --out-dir "${OUT}"
  rc=$?
  if (( rc == 0 )); then
    date -Is > "${OUT}/.done/${TAG}"
    echo "C6 DONE ${TAG} $((SECONDS-t0))s"
  else
    rc_all=1
    echo "C6 FAILED ${TAG} rc=${rc} $((SECONDS-t0))s"
  fi
done
echo "[c6] task ${SLURM_ARRAY_TASK_ID} finished rc=${rc_all} $(date -Is)"
exit ${rc_all}
