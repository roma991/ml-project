# Отчёт по подбору: `wine`

- **Дата:** 2026-10-06 23:38:24
- **Версия конфига:** 1.0.0
- **Задача:** multiclass_classification
- **Метрика подбора:** `accuracy` (maximize)
- **CV:** 5 фолдов
- **Бюджет:** n_trials=10, timeout=300s
- **Seed:** 42

## Итог

- **Победитель:** `RandomForestClassifier`
- **CV-метрика победителя:** 0.9717
- **Baseline (default params):** 0.9786
- **Метрика на тесте:** 1.0000

## Статистика испытаний

- Завершено: **10**
- Упало: **0**
- Отсечено (pruned): **0**

## Лидерборд (топ-10)

| # | value | GradientBoostingClassifier__max_depth | GradientBoostingClassifier__n_estimators | RandomForestClassifier__max_depth | RandomForestClassifier__min_samples_split | RandomForestClassifier__n_estimators | categorical_encode | model | numeric_impute | numeric_scale |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.9717 | nan | nan | 3.0 | 2.0 | 100.0 | onehot | RandomForestClassifier | median | minmax |
| 3 | 0.9717 | nan | nan | 3.0 | 2.0 | 128.0 | onehot | RandomForestClassifier | median | standard |
| 5 | 0.9717 | nan | nan | 5.0 | 3.0 | 64.0 | ordinal | RandomForestClassifier | mean | standard |
| 0 | 0.9645 | nan | nan | 7.0 | 2.0 | 128.0 | onehot | RandomForestClassifier | mean | standard |
| 8 | 0.9645 | nan | nan | 5.0 | 3.0 | 100.0 | onehot | RandomForestClassifier | mean | minmax |
| 4 | 0.9502 | 2.0 | 100.0 | nan | nan | nan | onehot | GradientBoostingClassifier | median | none |
| 6 | 0.9502 | 2.0 | 100.0 | nan | nan | nan | onehot | GradientBoostingClassifier | median | minmax |
| 7 | 0.9502 | 2.0 | 100.0 | nan | nan | nan | onehot | GradientBoostingClassifier | median | standard |
| 1 | 0.9081 | 5.0 | 100.0 | nan | nan | nan | ordinal | GradientBoostingClassifier | median | minmax |
| 9 | 0.9081 | 5.0 | 64.0 | nan | nan | nan | onehot | GradientBoostingClassifier | mean | standard |

## Важность гиперпараметров

| Параметр | Важность |
|---|---|
| `model` | 0.4514 |
| `RandomForestClassifier__max_depth` | 0.2607 |
| `RandomForestClassifier__n_estimators` | 0.1314 |
| `categorical_encode` | 0.1130 |
| `numeric_scale` | 0.0220 |
| `RandomForestClassifier__min_samples_split` | 0.0162 |
| `numeric_impute` | 0.0054 |
| `GradientBoostingClassifier__n_estimators` | 0.0000 |
| `GradientBoostingClassifier__max_depth` | 0.0000 |

## Кривая лучшего значения

```
trial   0: 0.9645
trial   1: 0.9645
trial   2: 0.9717
trial   3: 0.9717
trial   4: 0.9717
trial   5: 0.9717
trial   6: 0.9717
trial   7: 0.9717
trial   8: 0.9717
trial   9: 0.9717
```
