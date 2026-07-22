#!/bin/bash
# Feed the persistence-image sweep into the queue as capacity frees up.
#
# The account allows only 20 submitted tasks (MaxSubmitJobs=20) and 2 concurrent
# nodes, so a 25-configuration stage cannot be queued in one go. This submits
# whole configurations -- one array task trains all 8 seeds of one image set --
# because a partial cell would be ensembled over a different seed set than the
# configurations it is ranked against, which is the comparison the
# pre-registration forbids.
#
# Same shape as control_driver.sh, which did this for the 2x2 control.
#
#   nohup automl/slurm/pi_driver.sh <manifest.json> > automl/logs/pi_driver.log 2>&1 &
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
cd "${REPO}"

MANIFEST=${1:?usage: pi_driver.sh <manifest.json> [chunk]}
CHUNK=${2:-10}                 # tasks per submission
CAP=${CAP:-18}                 # stay under the 20-task account limit

N=$(python3 -c "import json;print(len(json.load(open('${MANIFEST}'))))")
echo "DRIVER manifest=${MANIFEST} configs=${N} chunk=${CHUNK} cap=${CAP}"

used() { squeue -u "$USER" -h -r -t PENDING,RUNNING,CONFIGURING 2>/dev/null | wc -l; }

i=0
while [ "$i" -lt "$N" ]; do
  hi=$(( i + CHUNK - 1 )); [ "$hi" -ge "$N" ] && hi=$(( N - 1 ))
  n=$(( hi - i + 1 ))
  while true; do
    u=$(used)
    if [ $(( CAP - u )) -ge "$n" ]; then
      out=$(sbatch --parsable --array="${i}-${hi}" \
            automl/slurm/pi_sweep.sh "${MANIFEST}" 2>&1)
      if [[ "$out" =~ ^[0-9]+$ ]]; then
        echo "SUBMITTED configs=${i}-${hi} tasks=${n} jobid=${out} (queue had ${u})"
        break
      fi
      echo "RETRY ${i}-${hi}: ${out##*error: }"
    fi
    sleep 45
  done
  i=$(( hi + 1 ))
  sleep 15
done
echo "DRIVER DONE all ${N} configurations submitted $(date -Is)"
