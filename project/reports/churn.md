# Отчёт по подбору: `churn`

- **Дата:** 2026-10-06 23:28:43
- **Версия конфига:** 1.0.0
- **Задача:** classification
- **Метрика подбора:** `roc_auc` (maximize)
- **CV:** 5 фолдов
- **Бюджет:** n_trials=30, timeout=900s
- **Seed:** 42

## Итог

- **Победитель:** `GradientBoostingClassifier`
- **CV-метрика победителя:** 0.8452
- **Baseline (default params):** 0.8180
- **Метрика на тесте:** 0.8141

## Статистика испытаний

- Завершено: **14**
- Упало: **0**
- Отсечено (pruned): **0**

## Лидерборд (топ-10)

| # | value | AdaBoostClassifier__n_estimators | GradientBoostingClassifier__max_depth | GradientBoostingClassifier__n_estimators | RandomForestClassifier__max_depth | RandomForestClassifier__min_samples_split | RandomForestClassifier__n_estimators | categorical_encode | model | numeric_impute | numeric_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 0.8452 | nan | 5.0 | 64.0 | nan | nan | nan | onehot | GradientBoostingClassifier | median | standard |
| 12 | 0.8440 | nan | 5.0 | 64.0 | nan | nan | nan | onehot | GradientBoostingClassifier | mean | none |
| 11 | 0.8438 | nan | 5.0 | 64.0 | nan | nan | nan | onehot | GradientBoostingClassifier | mean | none |
| 10 | 0.8437 | nan | 5.0 | 64.0 | nan | nan | nan | onehot | GradientBoostingClassifier | mean | none |
| 5 | 0.8425 | 100.0 | nan | nan | nan | nan | nan | onehot | AdaBoostClassifier | mean | minmax |
| 4 | 0.8424 | 128.0 | nan | nan | nan | nan | nan | onehot | AdaBoostClassifier | mean | minmax |
| 9 | 0.8424 | 128.0 | nan | nan | nan | nan | nan | onehot | AdaBoostClassifier | mean | standard |
| 7 | 0.8415 | 128.0 | nan | nan | nan | nan | nan | ordinal | AdaBoostClassifier | mean | minmax |
| 2 | 0.8414 | 100.0 | nan | nan | nan | nan | nan | ordinal | AdaBoostClassifier | mean | standard |
| 1 | 0.8413 | 100.0 | nan | nan | nan | nan | nan | ordinal | AdaBoostClassifier | mean | none |

## Важность гиперпараметров

| Параметр | Важность |
|---|---|
| `model` | 0.4922 |
| `numeric_impute` | 0.2837 |
| `numeric_scale` | 0.1124 |
| `GradientBoostingClassifier__max_depth` | 0.0444 |
| `GradientBoostingClassifier__n_estimators` | 0.0444 |
| `categorical_encode` | 0.0230 |
| `AdaBoostClassifier__n_estimators` | 0.0000 |
| `RandomForestClassifier__max_depth` | 0.0000 |
| `RandomForestClassifier__n_estimators` | 0.0000 |
| `RandomForestClassifier__min_samples_split` | 0.0000 |

## Кривая лучшего значения

```
trial   0: 0.8190
trial   1: 0.8413
trial   2: 0.8414
trial   3: 0.8414
trial   4: 0.8424
trial   5: 0.8425
trial   6: 0.8425
trial   7: 0.8425
trial   8: 0.8425
trial   9: 0.8425
trial  10: 0.8437
trial  11: 0.8438
trial  12: 0.8440
trial  13: 0.8452
```
