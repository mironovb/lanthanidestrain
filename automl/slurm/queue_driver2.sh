#!/bin/bash
# Second-stage submission driver: runs after the sweep stages have drained.
# Same adaptive sizing as queue_driver.sh, but it waits until the first driver
# has submitted everything and the queue has real capacity, so the shortlist
# re-run and the GPU-side work do not starve the sweeps.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
cd "${REPO}"

CAP=${CAP:-38}
MIN_SLOTS=${MIN_SLOTS:-4}

WORK=(
  "REPEATS=5 ROW_FILTER=has3d|automl/slurm/champion.sh|7"
  "REPEATS=5 ROW_FILTER=ok_only OUT_DIR=${REPO}/automl/artifacts/champion_okonly|automl/slurm/champion.sh|7"
)

used_tasks() { squeue -u "$USER" -h -r -t PENDING,RUNNING,CONFIGURING 2>/dev/null | wc -l; }

# Wait for the first driver to finish handing in its stages.
while pgrep -f "bash automl/slurm/queue_driver.sh" > /dev/null 2>&1; do sleep 120; done
echo "DRIVER2: stage-1 driver finished, waiting for capacity"

for entry in "${WORK[@]}"; do
  IFS='|' read -r envs script maxarr <<< "$entry"
  [ -f "$script" ] || { echo "SKIP missing $script"; continue; }
  while true; do
    n=$(used_tasks)
    slots=$(( CAP - n ))
    if [ "$slots" -ge "$MIN_SLOTS" ]; then
      arr=$(( slots < maxarr ? slots : maxarr ))
      out=$(env $envs sbatch --array=0-$((arr-1)) "$script" 2>&1)
      if [[ "$out" == Submitted* ]]; then
        echo "SUBMITTED $script array=0-$((arr-1)) -> ${out##* } (was using ${n} tasks)"
        break
      else
        echo "RETRY $script (slots=${slots}): ${out##*error: }"
      fi
    fi
    sleep 120
  done
  sleep 30
done
echo "QUEUE DRIVER 2: all stages submitted"
