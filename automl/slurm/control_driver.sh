#!/bin/bash
# Feeds the remaining cells of the control factorial as queue capacity frees up.
#
# The account allows 40 submitted tasks but only 2 concurrent nodes, so the
# whole 2x2 cannot be queued at once.  Unlike queue_driver.sh this submits
# *complete* arrays: a cell is 16 matched seeds and a partial cell would produce
# an ensemble over a different seed set than the arm it is a control for, which
# is precisely the comparison the pre-registration forbids.
#
# T0 and T1 are submitted by hand before this starts; this handles the rest.
# One line per submission -- that is what the monitor watches.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
cd "${REPO}"

CAP=${CAP:-38}                 # stay under the 40-task account limit

# Ordered by how much each answers.  T0w guards the capacity objection against
# T0; P1 is the objective-free topology test; S1 completes the SNN row.
WORK=("T0w:16" "P1:16" "P0:1" "S1:16")

used() { squeue -u "$USER" -h -r -t PENDING,RUNNING,CONFIGURING 2>/dev/null | wc -l; }

for entry in "${WORK[@]}"; do
  cell="${entry%%:*}"; n="${entry##*:}"
  while true; do
    u=$(used)
    if [ $(( CAP - u )) -ge "$n" ]; then
      out=$(CELL="$cell" sbatch --parsable --array=0-$((n-1)) \
            automl/slurm/topo_control.sh 2>&1)
      if [[ "$out" =~ ^[0-9]+$ ]]; then
        echo "SUBMITTED cell=${cell} tasks=${n} jobid=${out} (queue had ${u})"
        break
      fi
      echo "RETRY cell=${cell}: ${out##*error: }"
    fi
    sleep 60
  done
  sleep 20
done
echo "CONTROL DRIVER: all cells submitted"
