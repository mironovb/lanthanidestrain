#!/bin/bash
# Keep the generated reports in step with whatever has landed.
# Cheap enough to run on the login node: it only reads JSONL/parquet results and
# re-renders tables and figures; it fits no models.
set -uo pipefail
REPO=/home/gridsan/bmironov/lanthanidestrain
source /etc/profile.d/modules.sh
module load anaconda/Python-ML-2025a
export PYTHONPATH="${REPO}:${PYTHONPATH:-}" PYTHONWARNINGS=ignore
cd "${REPO}"

python3 -m automl.analyze --top 20 > automl/reports/leaderboard.txt 2>&1
python3 -m automl.figures            >> automl/reports/leaderboard.txt 2>&1
# Primary ensemble: protocol-B folds only, so the stacking weights are
# interpretable.  The all-protocol version is kept as a secondary view.
python3 -m automl.ensemble --protocol-b-only --min-r2 0.30 --max-models 40 \
    > automl/reports/ensemble.txt 2>&1
python3 -m automl.ensemble --min-r2 0.30 --max-models 40 \
    --out-dir automl/reports/ensemble_allprotocols \
    > automl/reports/ensemble_all_protocols.txt 2>&1
python3 -m automl.compare --reference "baseline 2D, LightGBM" --n-boot 400 \
    > automl/reports/paired_comparison.txt 2>&1
python3 -m automl.compare --reference "baseline 2D, CatBoost + group wts" \
    --n-boot 400 --out automl/reports/paired_vs_catboost.csv \
    > automl/reports/paired_vs_catboost.txt 2>&1
python3 -m automl.compare --sweep-dir automl/artifacts/sweeps/ablation_catboost \
    --reference baseline_2d --n-boot 500 \
    --out automl/reports/paired_catboost_ablation.csv \
    > automl/reports/paired_catboost_ablation.txt 2>&1
python3 -m automl.compare --champ-dir automl/artifacts/champion_okonly \
    --reference "baseline 2D, CatBoost + group wts" --n-boot 400 \
    --out automl/reports/paired_okonly.csv \
    > automl/reports/paired_okonly.txt 2>&1
python3 -m automl.error_breakdown >> automl/reports/leaderboard.txt 2>&1
python3 -m automl.split_variability >> automl/reports/leaderboard.txt 2>&1
python3 -m automl.make_tables >> automl/reports/leaderboard.txt 2>&1
# --- topological arms (Stage 3-6) -------------------------------------------
# Regenerates the per-arm table, the paired bootstraps against BOTH baselines,
# and the blend analysis that establishes complementarity with CatBoost.
# Each writes its own report file; failures are logged rather than aborting the
# refresh, because these depend on GPU runs that may not be present.
python3 -m automl.topo.compare_arms --n-boot 200 \
    > automl/reports/topo_comparison.txt 2>&1 || \
    echo "topo compare_arms unavailable" >> automl/reports/topo_comparison.txt
python3 -m automl.topo.adjacent_test --n-boot 400 --baseline baseline::mlp::none \
    > automl/reports/adjacent_vs_fcnn.txt 2>&1 || \
    echo "adjacent_test vs FCNN unavailable" >> automl/reports/adjacent_vs_fcnn.txt
python3 -m automl.topo.adjacent_test --n-boot 400 --baseline baseline::catboost::none \
    > automl/reports/adjacent_vs_catboost.txt 2>&1 || \
    echo "adjacent_test vs CatBoost unavailable" >> automl/reports/adjacent_vs_catboost.txt
python3 -m automl.topo.ensemble_adjacent --n-boot 400 --baseline baseline::mlp::none \
    > automl/reports/adjacent_ensemble.txt 2>&1 || \
    echo "ensemble unavailable" >> automl/reports/adjacent_ensemble.txt
python3 -m automl.topo.blend_test --n-boot 400 \
    > automl/reports/adjacent_blend.txt 2>&1 || \
    echo "blend unavailable" >> automl/reports/adjacent_blend.txt
python3 -m automl.qc.reopt_report --solvents water,octanol \
    > automl/reports/reopt_audit.txt 2>&1 || \
    echo "reopt audit unavailable" >> automl/reports/reopt_audit.txt

n=$(python3 -c "
import pandas as pd
try:
    print(len(pd.read_csv('automl/reports/all_results.csv')))
except Exception:
    print(0)")
echo "REFRESH $(date -Is): ${n} experiments"
