#!/bin/bash
# Periodically regenerate the reports while the sweeps drain.
set -uo pipefail
cd /home/gridsan/bmironov/lanthanidestrain
for i in $(seq 1 200); do
  bash automl/refresh_reports.sh >> automl/logs/refresh.log 2>&1
  n=$(squeue -u "$USER" -h -r -t PENDING,RUNNING 2>/dev/null | wc -l)
  echo "$(date -Is) refresh #$i, queue depth $n" >> automl/logs/refresh.log
  if [ "$n" -eq 0 ]; then
    echo "$(date -Is) queue empty, final refresh done" >> automl/logs/refresh.log
    break
  fi
  sleep 900
done
