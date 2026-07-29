#!/bin/bash
# Drip-feed submitter: keep a campaign under the account's submit cap.
#
# This account is capped at MaxSubmitJobs=40 per partition AND GrpTRES node=1 on
# xeon-g6-volta (node=2 on xeon-p8).  Array tasks count individually against
# MaxSubmit, so `sbatch --array=0-47` on top of a queue that already holds 25
# tasks is rejected outright:
#
#   sbatch: error: Batch job submission failed: Job violates accounting/QOS
#   policy (job submit limit, user's size and/or time limits)
#
# Generalises control_driver.sh / pi_driver.sh, which each hard-coded one
# campaign.  Runs on the LOGIN NODE -- it only polls squeue and calls sbatch,
# which AGENTS.md explicitly permits.
#
# Usage:
#   automl/slurm/campaign_driver.sh <script> <n_tasks> [chunk] [cap] [START=n] [env=val ...]
#
# START=n resumes from task n.  Resumption is the normal case, not an edge
# case: the driver is killed and restarted whenever the cap or a script
# changes, and without it a restart re-submits work that is already queued.
#
# Examples:
#   automl/slurm/campaign_driver.sh automl/slurm/topo_objective.sh 48
#   automl/slurm/campaign_driver.sh automl/slurm/topo_encoder.sh 16 8 38 ARM=D0
#
# Resumable: it submits index ranges, so re-running after a kill continues from
# whatever is left rather than restarting.  Nothing is hidden -- it prints every
# submission and every wait, and exits non-zero if it could not place them all.
set -uo pipefail

SCRIPT="${1:?usage: campaign_driver.sh <script> <n_tasks> [chunk] [cap] [ENV=VAL ...]}"
NTASKS="${2:?number of array tasks}"
CHUNK="${3:-8}"
CAP="${4:-34}"
shift 4 2>/dev/null || shift $#
EXPORTS=()
START=0
for kv in "$@"; do
  if [[ "$kv" == START=* ]]; then START="${kv#START=}"; else EXPORTS+=("$kv"); fi
done

if [[ ! -f "${SCRIPT}" ]]; then
  echo "no such script: ${SCRIPT}" >&2; exit 2
fi
PART=$(grep -m1 -oP '(?<=--partition=)\S+' "${SCRIPT}" || echo "")
NAME=$(grep -m1 -oP '(?<=--job-name=)\S+' "${SCRIPT}" || echo "")
echo "[driver] ${SCRIPT}  tasks=0-$((NTASKS - 1))  chunk=${CHUNK}  cap=${CAP}"
echo "[driver] partition=${PART:-?}  jobname=${NAME:-?}  extra env: ${EXPORTS[*]:-none}"

mine() {
  # TASKS in flight, not squeue lines.  A pending array collapses to a single
  # row like "5278104_[9-15]", so counting lines reports 1 where the cap sees 7
  # -- which made the first version of this driver submit straight into an
  # AssocMaxSubmitJobLimit rejection.  -r expands array elements to one row each,
  # which is the unit MaxSubmitJobs actually counts.
  squeue -u "$USER" -h -r -p "${PART}" -t PD,R,CG -o "%i" 2>/dev/null | wc -l
}

submitted=0
next=${START}
(( START > 0 )) && echo "[driver] resuming at task ${START}"
while (( next < NTASKS )); do
  n=$(mine)
  room=$(( CAP - n ))
  if (( room < 1 )); then
    echo "[driver] ${n} in flight on ${PART}, cap ${CAP} -- waiting 120s"
    sleep 120
    continue
  fi
  (( room > CHUNK )) && room=${CHUNK}
  last=$(( next + room - 1 ))
  (( last > NTASKS - 1 )) && last=$(( NTASKS - 1 ))

  env_args=()
  for kv in "${EXPORTS[@]:-}"; do [[ -n "$kv" ]] && env_args+=("$kv"); done
  if (( ${#env_args[@]} )); then
    out=$(env "${env_args[@]}" sbatch --array="${next}-${last}" "${SCRIPT}" 2>&1)
  else
    out=$(sbatch --array="${next}-${last}" "${SCRIPT}" 2>&1)
  fi
  rc=$?
  if (( rc != 0 )); then
    # Almost always the submit cap racing us; back off rather than give up.
    echo "[driver] submit ${next}-${last} refused (${out}); retrying in 120s"
    sleep 120
    continue
  fi
  echo "[driver] ${out}  (tasks ${next}-${last})"
  submitted=$(( submitted + last - next + 1 ))
  next=$(( last + 1 ))
  sleep 5
done

echo "[driver] placed ${submitted} tasks (indices ${START}-$((NTASKS - 1)))."
# Submitted is not finished, and the two are different states.
echo "[driver] NOTE: all tasks are SUBMITTED, not completed. Check with"
echo "         squeue -u \$USER  and the per-run outputs before reading results."
(( submitted == NTASKS - START )) || exit 1
