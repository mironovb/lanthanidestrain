# Как получены заявленные метрики — инструкция по воспроизведению (2026-08-18)

Документ отвечает на вопрос «откуда взялись числа» для основных метрик проекта и даёт
два пути их проверить: **быстрый** (пересчёт по закоммиченным OOF-предсказаниям, минуты)
и **полный** (переобучение, ~1.5 ч на 8 CPU). Все ссылки — на файлы этого репозитория,
ветка `descriptor-arm-metal-site`. Датасет и все артефакты запусков **уже в git**,
отдельно ничего скачивать не нужно.

---

## 0. Какие числа имеются в виду

| Группа | Число | Откуда (артефакт) |
|---|---|---|
| **A. Чемпион на невиданных экстрагентах** (основной режим) | A2: equal-extractant macro MAE **0.3192**, pooled MAE 0.4189, pooled R² 0.323, sign acc 0.847; A2+TP: macro **0.3175**; PAIRMEAN_baseline: macro 0.4482 / pooled MAE 0.3825 / R² 0.444 | `runs/gen4_candidates_20260817T003449Z/leaderboard.csv`, `per_seed_metrics.csv`, `paired_bootstrap.csv`; те же A2-числа в `runs/gen3_primary_20260815T181555Z/aggregate/aggregate_metrics.csv` (арм `A2_current_champion`) |
| **B. R² по режимам** | unseen extractant: pooled R² **0.31** (oracle-offset потолок 0.51, oracle-affine 0.58; 5/10-shot offset 0.41/0.46); seen extractant / unseen conditions: pooled R² **0.744**, macro MAE 0.249 | `runs/ablation_all_20260810T181058Z/aggregate/r2_supplement/{arm_r2,calibration_bounds}.csv`; `runs/condition_regime_20260815T171535Z/summary.json` |
| **C. k-shot калибровка** | A2+TP `cross_condition`: 0.3785 → 0.2413 при k=10; PAIRMEAN 0.2322; `y≈b·ΔZ` без модели 0.2395; `random`: 0.3327 → 0.1757 | `runs/kshot_calibration_20260817T181016Z/{head_to_head_vs_nulls,grid_metrics_seedmean,decision_table}.csv`; описание `docs/kshot_calibration_study_20260817.md` |

Все MAE / R² — в единицах **log₁₀ SF** (см. §2). «macro MAE» везде означает
*equal-extractant macro MAE* — среднее по экстрагентам от MAE внутри экстрагента.

> ⚠️ В `docs/gen4_candidate_study_20260816.md` таблица относится к *другому* прогону
> (`runs/gen4_candidates_confirm_5seeds/`, A2_refit 0.3188 / A2_refit_TP 0.3172). Числа
> выше — из кластерной репликации `gen4_candidates_20260817T003449Z`. Расхождение
> ±0.0005 — это шум порядка обхода потоков в лесах (см. §8), не ошибка.

---

## 1. Данные

* Единственный входной файл для группы A и B: `dataset with 3D structures/dataset.parquet`
  (2.7 MB, 5 992 строки × 2 261 колонка).
  SHA-256 `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd`
  (зафиксирован в `gen3_protocol.json` → `immutable_inputs.dataset_sha256`; gen3-раннер
  отказывается работать при другом хеше).
* Дополнительно только для арм `*_lig2d` в gen4: `dataset with 3D structures/ligand_2d_descriptors.parquet`
  (206 колонок `lig2d__*` на 190 уникальных SMILES; собран офлайн `scripts/build_ligand_descriptors.py`, RDKit).
  **Чемпион A2 / A2+TP его не использует.**
* **Не используются** для этих метрик: `features/*`, `geometries/*`, `sample_weights.csv`,
  `row_geometry_map.csv`, `dataset_geometry_available.parquet` (они нужны только 3D/SNN-веткам).
* Колонки `dataset.parquet`, которые реально участвуют (`src/lanthanide_separation/pairs.py:191-249, 444-450`):
  * `log_D` — целевая на уровне строки (десятичный логарифм коэффициента распределения);
  * `canonical_smiles` — идентификатор экстрагента и группа для CV (`EXTRACTANT_COLUMN`, `pairs.py:127`);
    `extractant_name`, `extractant_group` — только для аудита/карантина;
  * `metal` (15 лантанидов), `Ionic Radius_metal`;
  * 64 числовые `cond__*` — экспериментальные условия;
  * `geometry_ok` — фильтр пар (обе стороны должны быть `True`);
  * 10 RDKit-дескрипторов `MolWt, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds,
    NumAromaticRings, NumAliphaticRings, RingCount, FractionCSP3, MolLogP` и **2 048 готовых бит `ecfp_*`**
    (ECFP **не считается на лету** — читается из parquet; RDKit для A2 не нужен).

Минимальная загрузка:

```python
import pandas as pd
df = pd.read_parquet("dataset with 3D structures/dataset.parquet")
```

---

## 2. Когорта пар: 5 992 строки → 6 699 пар / 34 экстрагента

Функция `build_lanthanide_pair_dataset` (`src/lanthanide_separation/pairs.py:541`). Все числа
ниже — из `runs/gen3_primary_20260815T181555Z/run_0_split_104729_model_42/pair_build_audit.json`.

1. 5 992 исходных строк.
2. Карантин **−129**: строки с SMILES TODGA, но `extractant_name != "TODGA"` (`pairs.py:119,264-265`).
3. Только лантаниды и конечный `log_D` (0 удалено).
4. **«Все 64 условия записаны»**: строки с любым NaN в `cond__*` выбрасываются, **−2 851** → 3 012 строк
   (`pairs.py:674-676`). Это главный фильтр, из-за него в когорте 34 экстрагента, а не 190.
5. Ячейка = (`canonical_smiles`, все 64 `cond__*`, `metal`); числовые поля агрегируются медианой,
   `n_replicates` = размер группы (`pairs.py:717-786`) → 2 520 ячеек, 198 реплицированных.
6. Пары формируются **внутри группы (`canonical_smiles`, все 64 `cond__*`)** = точное совпадение
   условий; `pair_scope="all"` → все неупорядоченные пары металлов, **A всегда легче B**
   (`Z_A < Z_B`, `pairs.py:808-813`) → 8 195 пар-кандидатов.
7. Фильтры пар: обе геометрии `geometry_ok` (**−688**), `replicate_policy="unique"` — обе ячейки
   измерены ровно один раз (**−802**), полный набор выбранных 3D-признаков (**−6**) →
   **6 699 пар, 34 экстрагента**, 28 exact-ECFP кластеров.

Целевая и метки:

```text
log_SF_A_over_B = log_D(A) − log_D(B)          # log10; SF = 10 ** log_SF  (pairs.py:869,884)
pair_label      = f"{metal_A}-{metal_B}"        # напр. "La-Ce"; страта CV и ключ PAIRMEAN
condition_id    = sha256(extractant + 64 cond)[:20]
pair_id         = sha256(condition_id, metal_A, metal_B)[:20]
cohort_sha256   = af3d91b718d0497818db52be8d93ecdee8c6c80eb8fc0aeda1c25c2c3f5e5f0e
```

Воспроизвести когорту (без обучения):

```python
from lanthanide_separation.pairs import build_lanthanide_pair_dataset
pair_data = build_lanthanide_pair_dataset(
    df, pair_scope="all", require_geometry=True, replicate_policy="unique",
    quarantine_known_bad=True, require_complete_conditions=True,
    delta3d_feature_set="compact-invariant",
)
frame = pair_data.frame            # 6699 × …, есть pair_id, extractant, pair_label, log_SF_A_over_B

from lanthanide_separation.feature_registry import build_feature_registry
a2_columns = build_feature_registry(pair_data).ablation_columns("A2")   # 2130 колонок
```

Проверено 2026-08-18 (≈20 с на ноутбуке): `frame.shape == (6699, 2191)`, 34 экстрагента,
274 `condition_id`, 91 `pair_label`, `len(a2_columns) == 2130`.

---

## 3. Разбиения (CV)

* Группа = **`extractant`** (`canonical_smiles`), не ECFP-кластер (`gen3_protocol.json:42`).
* Внешний CV: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)`,
  страта = `pair_label`, группы = extractant (`src/lanthanide_separation/evaluation.py:342-368`).
  Разбиение не зависит от таргета.
* Внутренний CV (подбор гиперпараметров): тот же сплиттер, `n_splits=3`,
  `random_state = split_seed + outer_fold*10_007` (`gen3_evaluation.py:805-812`).
* Split seeds: **104729, 130363, 155921, 196613, 262147**; model seed **42**.
* Проверки утечки: пересечение train/test по extractant, `source_id`, `geometry_key` = 0
  (`gen3_evaluation.py:764-780`).
* **gen4 не пересплитывает**: фолды берутся из gen3 `oof_predictions.csv` по `pair_id`
  (`scripts/run_gen4_candidates.py:181-190`); фактические назначения — в
  `runs/gen3_primary_20260815T181555Z/run_*_split_<seed>_model_42/split_assignments.csv`.

---

## 4. Модель A2 (чемпион) и A2_refit

**Признаки** `A2 = CONDITIONS + LN + 2D` = **2 130 колонок** (`feature_registry.py:114, 412-440`):

| блок | колонки | n |
|---|---|---|
| CONDITIONS | `base__cond__*` | 64 |
| LN | `pair__Z_A, pair__Z_B, pair__Z_mean, pair__delta_Z, pair__ionic_radius_A/B/mean, pair__delta_ionic_radius` | 8 |
| 2D | `base__MolWt … base__MolLogP` (10) + `base__ecfp_0..2047` | 2 058 |

**Оценщик** `AntisymmetricExtraTreesRegressor` (`evaluation.py:104-193`):
`Pipeline[SimpleImputer(median, add_indicator=True) → ExtraTreesRegressor(n_estimators=200)]`.

* Антисимметризация: к train добавляется «перевёрнутая» копия (`reverse_pair_features`,
  `pairs.py:1207-1252`: меняются местами `*_A/*_B`, знак у `pair__delta_*`), таргет с минусом;
  предсказание `(f(A,B) − f(B,A))/2` ⇒ `f(B,A) = −f(A,B)` точно.
* Веса выборки — group-balanced по экстрагенту: `w_i = n_rows / (n_groups · count[extractant_i])`
  (`evaluation.py:90-101`). Из-за них модель оптимизирует именно equal-extractant цель.
* Гиперпараметры на каждый внешний фолд выбираются внутренним 3-fold CV по сетке
  `(max_features, min_samples_leaf) ∈ {(0.35,2), (0.70,2), (1.00,4)}` (`evaluation.py:22-26`),
  критерий — inner equal-extractant macro MAE (`gen3_evaluation.py:62-127`).
* Seed внешнего фита: `model_seed + outer_fold*1009 + 9_999_991` (`gen3_evaluation.py:843`).

**A2_refit (gen4)** = тот же класс, те же 2 130 колонок, 200 деревьев, тот же seed, но
`max_features/min_samples_leaf` **не подбираются заново**, а читаются из
`inner_cv_tuning.csv` gen3-прогона (строки `arm == "A2_current_champion" & selected`,
`run_gen4_candidates.py:114-123`). В прогоне `gen4_candidates_20260817T003449Z` A2_refit
совпал с исходными предсказаниями gen3 (`A2_run`) до 1e-16.

---

## 5. Transitive projection (TP)

`transitive_projection` (`src/lanthanide_separation/gen4_candidates.py:74-114`). Внутри каждой
ячейки (`extractant`, `condition_id`) строится матрица инцидентности `M` (±1 по металлам пары),
решается `lstsq(M, p)` → пер-металльные скоры `s`, предсказания заменяются на `M @ s`,
т.е. `ŷ(A,B) = s_A − s_B`. Используются **только предсказания, не метки** — это бесплатный
пост-процессинг. Ячейки с < 2 строками не трогаются. Применяется ко всем армам:
`prediction_<arm>_TP`.

---

## 6. Определения метрик (все в `src/lanthanide_separation/gen3_metrics.py`)

Пусть `y` — `log_SF_A_over_B`, `p` — предсказание, `e` — экстрагент строки.

| метрика | формула | код |
|---|---|---|
| **equal_extractant_macro_mae** (главная) | `mean_e( mean_{i∈e} |y_i − p_i| )` — MAE внутри каждого экстрагента, затем **невзвешенное** среднее по 34 экстрагентам | `gen3_metrics.py:29-43, 118` |
| pooled_micro_mae | `mean_i |y_i − p_i|` по всем 6 699 строкам | `:119` |
| pooled_r2 | `sklearn.r2_score(y, p)` по всем строкам | `:130` |
| sign_accuracy | `mean( sign(y) == sign(p) )` | `:159-161` |
| per-extractant R² (guarded) | `r2_score` внутри экстрагента; **NaN, если < 8 строк или std(y) < 0.1**; из NaN считаются `equal_extractant_macro_r2` (mean) и `median_extractant_r2`; выживших 31 из 34 | `:19-26, 108-140` |
| worst_quartile_extractant_mae | среднее по худшим `ceil(0.25·34)=9` экстрагентам | `:106-107, 121-123` |
| adjacent / nonadjacent_ln_mae | MAE по парам с `Z_B − Z_A == 1` и остальным | `:97-100, 149-158` |
| prediction_dispersion_ratio | `std(p)/std(y)` | `:145-148` |
| PAIRMEAN_baseline | leave-fold-out: по train-фолду среднее `y` внутри (`pair_label`, extractant), затем среднее по экстрагентам → таблица по `pair_label`; без признаков лиганда | `run_gen4_candidates.py:126-144` |
| within_pair_r2 (gen4) | R² после вычитания средних по (`outer_fold`, `pair_label`) из `y` и `p` | `run_gen4_candidates.py:147-154` |

**Агрегация по seed'ам.** Метрики считаются **отдельно для каждого split seed** (6 699 строк),
затем в `leaderboard.csv` берётся **среднее по 5 seed'ам** и `macro_mae_split_sd` = std по seed'ам
(`run_gen4_candidates.py:315-364`). Пер-seed значения — `per_seed_metrics.csv`.

**Парный бутстрэп** (`gen3_metrics.py:182-245`): единица ресэмплинга — экстрагент;
статистика — среднее по экстрагентам от `(MAE_reference − MAE_candidate)` (положительно ⇒
кандидат лучше); `rng = np.random.default_rng(8675309)`, `rng.integers(0, 34, size=34)`,
10 000 реплик; CI95 = квантили 2.5/97.5 %. В gen4 бутстрэп идёт по **объединённым 5 seed'ам**
(33 495 строк). Правило решения в `scripts/gen4_decision.py`: бутстрэп только на 4
подтверждающих seed'ах, двусторонний p, Holm по замороженному набору кандидатов, проход только при
`mean Δ>0 ∧ ≥3/4 seeds ∧ ci95_low>0 ∧ holm_p<0.05`.

Ожидаемые числа (`gen4_candidates_20260817T003449Z`):

| arm | macro MAE (5-seed mean ± sd) | per-seed macro MAE (104729 / 130363 / 155921 / 196613 / 262147) | pooled MAE | pooled R² | sign acc |
|---|---|---|---|---|---|
| A2_refit | 0.319221 ± 0.007794 | 0.316442 / 0.310345 / 0.315668 / 0.323065 / 0.330582 | 0.418910 | 0.323261 | 0.847052 |
| A2_refit_TP | 0.317464 ± 0.007911 | 0.316172 / 0.307798 / 0.313212 / 0.321788 / 0.328349 | 0.418671 | 0.326879 | 0.846335 |
| PAIRMEAN_baseline | 0.448181 ± 0.003914 | 0.453041 / 0.450714 / 0.443807 / 0.448616 / 0.444730 | 0.382517 | 0.444487 | 0.849858 |

Бутстрэп `A2_refit_vs_A2_refit_TP`: Δ = +0.001757, CI95 [+0.0000291, +0.003616], p_better 0.9774,
улучшено 21/34 экстрагентов. `A2_refit_vs_PAIRMEAN_baseline`: Δ = −0.128961, CI [−0.1926, −0.0661].

Обратите внимание на **инверсию pooled vs macro**: PAIRMEAN лучше по pooled MAE / R²
(0.3825 / 0.444 против 0.4189 / 0.323 у A2), но катастрофически хуже по macro (0.448 против 0.319).
Причина — TODGA (43.6 % строк, всегда один в фолде). Поэтому pooled-статистики никогда не
используются для отбора (`gen3_protocol.json`), а любое «R² = X» без указания режима бессмысленно.

---

## 7. Два пути воспроизведения

### 7.1 Быстрый: пересчёт по закоммиченным OOF (без обучения, < 1 мин)

`scripts/reproduce_headline_metrics.py` — независимая реализация формул из §6 на голых
pandas/numpy (без импортов из пакета), чтобы формулы можно было прочитать глазами:

```bash
python scripts/reproduce_headline_metrics.py runs/gen4_candidates_20260817T003449Z/oof_predictions.csv --arms A2_refit A2_refit_TP PAIRMEAN_baseline
```

Проверено 2026-08-18: расхождение с `leaderboard.csv` по 9 метрикам ≤ 1.7e-16; бутстрэп
совпадает с `paired_bootstrap.csv` (Δ +0.001757, CI [+0.0000291, +0.003616], p 0.9774, 21/34).

Структура `oof_predictions.csv` (33 495 строк = 5 seed × 6 699 пар): `pair_id, extractant,
extractant_family, condition_id, pair_label, metal_A, metal_B, pair__Z_A, pair__Z_B,
log_SF_A_over_B, outer_fold, split_seed, model_seed, prediction_<arm>[…]`. Все, что нужно
для любой своей метрики, там есть.

Проверка когорты и набора признаков без обучения — сниппет в §2.

### 7.2 Полный: переобучение

Окружение: Python ≥ 3.10, `pip install -e .` (numpy, pandas, pyarrow, scikit-learn, scipy, joblib).
Для gen3-раннера дополнительно `.[gen3]` (torch, catboost) и **строгий контракт версий** из
`gen3_protocol.json` (Linux x86_64, Python 3.11.11, sklearn 1.9.0, numpy 2.4.6, pandas 3.0.5, …):
не-`--quick` запуск с другими версиями **прерывается** (`scripts/run_gen3_benchmark.py:330-334`).

**Вариант (а) — самый простой, только чемпион и baseline** (без версионного контракта; фолды
и гиперпараметры берутся из закоммиченного gen3-прогона; ~20–30 с на арм-фолд, весь набор
из 8 арм × 5 seed занял 1 ч 25 мин на 8 CPU):

```bash
python scripts/run_gen4_candidates.py --run-dir runs/gen3_primary_20260815T181555Z --split-seeds 104729 130363 155921 196613 262147 --arms A2_refit --n-estimators 200 --n-jobs 8 --bootstrap-replicates 10000 --output-dir runs/repro_gen4_$(date -u +%Y%m%dT%H%M%SZ)
```

Смоук-тест (30 деревьев, один seed, минуты; числа НЕ научные):

```bash
python scripts/run_gen4_candidates.py --run-dir runs/gen3_primary_20260815T181555Z --arms A2_refit --quick --output-dir runs/repro_gen4_quick
```

На выходе `leaderboard.csv`, `per_seed_metrics.csv`, `paired_bootstrap.csv`, `oof_predictions.csv`
с теми же колонками, что в §7.1 (`PAIRMEAN_baseline` и `*_TP` добавляются автоматически).

**Вариант (б) — полный gen3-протокол** (внутренний CV, все H1–H3 армы, ~60–90 мин на seed
при 8–16 CPU, 96 G в SLURM-обёртке):

```bash
python scripts/run_gen3_benchmark.py --protocol gen3_protocol.json --output-dir runs/repro_gen3/run_0_split_104729_model_42 --split-seed 104729 --model-seed 42 --n-jobs 16
```

и так для 5 seed'ов, затем `python scripts/aggregate_gen3_runs.py` (реальные вызовы — в
`slurm/gen3_multisplit.slurm`, `slurm/submit_gen3_multisplit.sh`, `docs/gen3_runbook.md`).

**Режим «известный экстрагент / новые условия»** (группа B, R² 0.744; 5 фитов ExtraTrees × 200 деревьев на 2 130 колонках, `n_jobs=-1` — минуты на многоядерной машине):

```bash
python scripts/evaluate_condition_regime.py
```

Та же когорта, те же 2 130 колонок и веса, но `GroupKFold(5)` по `condition_id` (274 ячейки).
Ожидается `summary.json`: pooled R² 0.7436, pooled MAE 0.2333, macro MAE 0.2489.

**R²-supplement и потолки калибровки** (группа B, R² 0.31 / oracle 0.51 / 0.58) — считаются
по агрегату gen2-абляции (та же когорта 6 699 пар, арм `A2`, split 104729 × 5 model seeds):

```bash
python scripts/analyze_oof_r2.py runs/ablation_all_20260810T181058Z/aggregate/cross_seed_oof_predictions.csv --calibration-arm A2
```

**k-shot** (группа C; пост-хок по OOF gen4, без обучения; сам цикл ~1 мин, запись артефактов ещё несколько):

```bash
python scripts/run_kshot_calibration.py runs/gen4_candidates_20260817T003449Z/oof_predictions.csv --arms A2_refit_TP A2_refit PAIRMEAN_baseline --n-draws 20
```

Протокол и определения (формы калибратора, политики, стратификация на «свободные» строки) —
`docs/kshot_calibration_study_20260817.md`.

---

## 8. На что напороться при воспроизведении

1. **Хеш датасета и версии.** gen3-раннер сверяет SHA-256 `dataset.parquet` и версии библиотек;
   при несовпадении падает. `run_gen4_candidates.py` и `evaluate_condition_regime.py` таких
   ворот не имеют — начинать лучше с них.
2. **Недетерминизм ±0.002 macro.** ExtraTrees с `n_jobs>1` даёт различия ~1e-16 в порядке
   суммирования, которые в *зависимых* армах (где предсказания подаются дальше) переворачивали
   сплиты; для чемпиона расхождения между машинами ожидаемы на уровне 3–4 знака после запятой.
   Сравнивать нужно с CI бутстрэпа, а не побитно.
3. **`macro` ≠ `pooled`.** Всегда указывать, какая MAE/R² имеется в виду; pooled-числа
   искажены TODGA (43.6 % строк).
4. **R² зависит от режима.** Одна и та же модель: 0.31 (новые экстрагенты) — 0.74 (новые
   условия). Per-extractant R² без ограждений уходит в −269 на почти константных экстрагентах.
5. **Sign accuracy 0.847 < 0.853 у «всегда отрицательного» предиктора** — не является
   свидетельством качества.
6. **Единицы.** Всё в log₁₀; `SF = 10 ** log_SF`. Шумовой пол по репликам ≈ 0.19–0.24 log.
7. `docs/gen4_candidate_study_20260816.md` цитирует прогон `gen4_candidates_confirm_5seeds`,
   а не `gen4_candidates_20260817T003449Z` — цифры отличаются в 4-м знаке.

---

## 9. Карта файлов

| Что | Где |
|---|---|
| Данные | `dataset with 3D structures/dataset.parquet` (+ `ligand_2d_descriptors.parquet` для lig2d-арм) |
| Когорта, пары, антисимметризация признаков | `src/lanthanide_separation/pairs.py` |
| Реестр признаковых блоков (A2 = 2 130 колонок) | `src/lanthanide_separation/feature_registry.py` |
| Модель, CV-сплиттер, веса | `src/lanthanide_separation/evaluation.py`, `gen3_evaluation.py` |
| Метрики и бутстрэп | `src/lanthanide_separation/gen3_metrics.py` |
| TP, gen4-армы | `src/lanthanide_separation/gen4_candidates.py`, `scripts/run_gen4_candidates.py`, `scripts/gen4_decision.py` |
| Замороженный протокол gen3 | `gen3_protocol.json` |
| Прогоны с числами | `runs/gen3_primary_20260815T181555Z/aggregate/`, `runs/gen4_candidates_20260817T003449Z/`, `runs/condition_regime_20260815T171535Z/`, `runs/ablation_all_20260810T181058Z/aggregate/r2_supplement/`, `runs/kshot_calibration_20260817T181016Z/` |
| Независимый пересчёт метрик | `scripts/reproduce_headline_metrics.py` |
