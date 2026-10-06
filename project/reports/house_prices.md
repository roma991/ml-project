# Отчёт по подбору: `house_prices`

- **Дата:** 2026-10-06 23:32:46
- **Версия конфига:** 1.0.0
- **Задача:** regression
- **Метрика подбора:** `neg_root_mean_squared_error` (maximize)
- **CV:** 5 фолдов
- **Бюджет:** n_trials=30, timeout=900s
- **Seed:** 42

## Итог

- **Победитель:** `GradientBoostingRegressor`
- **CV-метрика победителя:** -27879.8139
- **Baseline (default params):** -28300.7245
- **Метрика на тесте:** 0.9138

## Статистика испытаний

- Завершено: **30**
- Упало: **0**
- Отсечено (pruned): **0**

## Лидерборд (топ-10)

| # | value | GradientBoostingRegressor__learning_rate | GradientBoostingRegressor__max_depth | GradientBoostingRegressor__n_estimators | Ridge__alpha | categorical_encode | model | numeric_impute | numeric_scale |
|---|---|---|---|---|---|---|---|---|---|
| 22 | -27879.8139 | 0.1 | 3.0 | 300.0 | nan | onehot | GradientBoostingRegressor | median | standard |
| 21 | -28057.6285 | 0.1 | 3.0 | 300.0 | nan | onehot | GradientBoostingRegressor | median | standard |
| 23 | -28067.8553 | 0.1 | 3.0 | 300.0 | nan | onehot | GradientBoostingRegressor | median | standard |
| 2 | -28397.9489 | 0.1 | 3.0 | 100.0 | nan | onehot | GradientBoostingRegressor | median | standard |
| 29 | -28419.2110 | 0.1 | 3.0 | 300.0 | nan | onehot | GradientBoostingRegressor | mean | standard |
| 18 | -28613.7680 | 0.1 | 3.0 | 300.0 | nan | onehot | GradientBoostingRegressor | median | standard |
| 27 | -28644.6632 | 0.05 | 2.0 | 300.0 | nan | onehot | GradientBoostingRegressor | median | none |
| 16 | -28709.7286 | 0.1 | 3.0 | 100.0 | nan | onehot | GradientBoostingRegressor | median | standard |
| 10 | -28760.9735 | 0.1 | 3.0 | 100.0 | nan | onehot | GradientBoostingRegressor | median | standard |
| 11 | -28861.9774 | 0.1 | 3.0 | 100.0 | nan | onehot | GradientBoostingRegressor | median | standard |

## Важность гиперпараметров

| Параметр | Важность |
|---|---|
| `GradientBoostingRegressor__n_estimators` | 0.3986 |
| `numeric_impute` | 0.1864 |
| `numeric_scale` | 0.1765 |
| `categorical_encode` | 0.1174 |
| `model` | 0.1174 |
| `GradientBoostingRegressor__learning_rate` | 0.0019 |
| `GradientBoostingRegressor__max_depth` | 0.0019 |
| `Ridge__alpha` | 0.0000 |

## Кривая лучшего значения

```
trial   0: -31191.6584
trial   1: -31191.6584
trial   2: -28397.9489
trial   3: -28397.9489
trial   4: -28397.9489
trial   5: -28397.9489
trial   6: -28397.9489
trial   7: -28397.9489
trial   8: -28397.9489
trial   9: -28397.9489
trial  10: -28397.9489
trial  11: -28397.9489
trial  12: -28397.9489
trial  13: -28397.9489
trial  14: -28397.9489
trial  15: -28397.9489
trial  16: -28397.9489
trial  17: -28397.9489
trial  18: -28397.9489
trial  19: -28397.9489
trial  20: -28397.9489
trial  21: -28057.6285
trial  22: -27879.8139
trial  23: -27879.8139
trial  24: -27879.8139
trial  25: -27879.8139
trial  26: -27879.8139
trial  27: -27879.8139
trial  28: -27879.8139
trial  29: -27879.8139
```
