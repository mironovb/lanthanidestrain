#!/bin/bash
#SBATCH --job-name=ln_chmiss
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=11:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/gridsan/bmironov/lanthanidestrain/automl/logs/chmiss_%A_%a.out
#SBATCH --error=/home/gridsan/bmironov/lanthanidestrain/automl/logs/chmiss_%A_%a.err

# The champion array was sharded before the shortlist was extended with the
# best-of-breed stack, so six configurations were never assigned to a shard.
# This job runs exactly the ones whose result JSON is absent -- it is
# idempotent, so re-running it is safe.
set -euo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

python3 - "${SLURM_ARRAY_TASK_ID}" "${SLURM_ARRAY_TASK_COUNT}" <<'PY'
import json, pathlib, sys
import automl.champion as champ

shard, nshards = int(sys.argv[1]), int(sys.argv[2])
out = pathlib.Path("automl/artifacts/champion")
done = {json.loads(j.read_text())["label"] for j in out.glob("champ_*.json")}
missing = [c for c in champ.SHORTLIST if c[0] not in done]
champ.SHORTLIST = missing
print(f"[chmiss] {len(missing)} missing configs; shard {shard}/{nshards}", flush=True)
sys.argv = ["champion", "--shard", str(shard), "--num-shards", str(nshards),
            "--repeats", "5", "--row-filter", "has3d",
            "--n-jobs", str(6)]
champ.main()
PY
echo "DONE chmiss ${SLURM_ARRAY_TASK_ID} $(date -Is)"
