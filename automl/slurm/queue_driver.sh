#!/bin/bash
# Autonomous submission driver.
#
# SuperCloud caps this account at 40 *submitted array tasks* and 2 concurrent
# nodes on xeon-p8, so the whole AutoML plan cannot be queued at once.  This
# driver waits for capacity and then submits the next stage with an array size
# that fits the free slots, so the pipeline advances without a human
# re-submitting.  One line is printed per submission; that is what the monitor
# watches.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
cd "${REPO}"

CAP=${CAP:-38}          # stay just under the 40-task account limit
MIN_SLOTS=${MIN_SLOTS:-4}

# Each entry: "ENV_ASSIGNMENTS|script|max_array_tasks"
WORK=(
  "REPEATS=3 ROW_FILTERS=has3d,ok_only|automl/slurm/sweep_ablation3.sh|10"
  "REPEATS=2|automl/slurm/select2.sh|6"
  "REPEATS=2|automl/slurm/sweep_models.sh|16"
  "REPEATS=2 N_TRIALS=45|automl/slurm/sweep_optuna.sh|16"
)

used_tasks() { squeue -u "$USER" -h -r -t PENDING,RUNNING,CONFIGURING 2>/dev/null | wc -l; }

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
    sleep 90
  done
  sleep 30
done
echo "QUEUE DRIVER: all stages submitted"
