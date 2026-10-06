# ML Pipeline с подбором гиперпараметров на Optuna

Универсальный пайплайн обучения и подбора гиперпараметров для задач
бинарной/мультиклассовой классификации и регрессии.

Один YAML-конфиг описывает данные, признаки, пространство поиска и бюджет.
Одна команда обучает модель и генерирует отчёт. Одна команда поднимает сервис
в Docker.

**Новый датасет = новый YAML. Ноль строк Python.**

---

## 1. Как запустить обучение

Из корня проекта `PythonProject/`:

```bash
python -m project.src.main --config project/config/churn.yaml
python -m project.src.main --config project/config/wine.yaml
python -m project.src.main --config project/config/house_prices.yaml
```

После запуска:
- артефакт сохраняется в `project/models/<config_name>/<ModelName>.joblib`
- отчёт генерируется в `project/reports/<config_name>.md`

### Требования

- Python 3.13
- Зависимости: `pip install -r project/requirements.txt`

### Что делает `main.py`

1. Загружает данные из `project/data/<path>`.
2. Убирает `drop_columns`, отделяет target.
3. Делит выборку на train/test по `dataset.split`.
4. Обучает **baseline** — первого кандидата с дефолтными параметрами.
5. Запускает **Optuna** с условным пространством поиска.
6. Восстанавливает победителя из `study.best_params`.
7. Считает метрику на тесте **один раз**.
8. Сохраняет артефакт: `{"model": pipeline, "metadata": {...}}`.
9. Генерирует `project/reports/<config_name>.md`.

---

## 2. Как поднять сервис в Docker

Из корня `PythonProject/`:

```bash
docker compose up --build
```

Сервис доступен на `http://127.0.0.1:8003/` (порт настраивается в `docker-compose.yml`).

Проверка:

```bash
curl http://127.0.0.1:8003/
curl http://127.0.0.1:8003/configs
curl http://127.0.0.1:8003/configs/churn
```

Остановить:

```bash
docker compose down
```

---

## 3. Как добавить новый датасет

1. Положи CSV в `project/data/<name>.csv`.
2. Создай `project/config/<name>.yaml` по схеме (см. раздел 4).
3. Запусти:
   ```bash
   python -m project.src.main --config project/config/<name>.yaml
   ```

**Никаких изменений в Python-коде не требуется.**

Чтобы добавить нового кандидата в существующий конфиг — тоже достаточно правки YAML: добавь ещё одну запись в `search_space.models` со своим `class` и `parameters`.

---

## 4. Схема конфига

```yaml
version: "1.0.0"
task: classification          # classification | regression
name: churn

dataset:
  path: data/churn.csv        # относительно project/
  target: Churn
  drop_columns: [customerID]
  split:
    test_size: 0.2
    random_state: 42
    stratify: true

features:
  numeric:
    - name: tenure
      dtype: int64
      min: 0
      max: 72
  categorial:
    - name: Contract
      dtype: object
      values: ["Month-to-month", "One year", "Two year"]

metric:
  primary: roc_auc
  additional: [accuracy, precision, recall, f1]

search:
  method: optuna
  cv: 5
  scoring: roc_auc            # sklearn-строка

search_space:
  preprocessing:
    numeric_impute:     { type: categorical, values: [median, mean] }
    numeric_scale:      { type: categorical, values: [standard, minmax, none] }
    categorical_encode: { type: categorical, values: [onehot, ordinal] }

  models:                     # список кандидатов
    - name: RandomForestClassifier
      class: sklearn.ensemble.RandomForestClassifier
      parameters:
        n_estimators: [64, 100, 128]       # grid-формат
        max_depth: [3, 5, 7]
    - name: GradientBoostingClassifier
      class: sklearn.ensemble.GradientBoostingClassifier
      parameters:
        n_estimators: [64, 100, 128]
        max_depth: [2, 3, 5]

tuning:
  metric: roc_auc
  direction: maximize         # maximize | minimize
  cv:
    folds: 5
    stratify: true
  budget:
    n_trials: 30
    timeout_seconds: 900
  seed: 42
```

### Важно

- `parameters` поддерживают **два формата**:
  - **grid** — список значений: `n_estimators: [64, 100, 128]`
  - **диапазон** — словарь: `{type: int, low: 50, high: 500, step: 50}` или `{type: float, low: 0.01, high: 0.3, log: true}`
- Пространство поиска **условное**: параметры сэмплируются только для выбранной модели. Параметр `C` не сэмплируется, если Optuna выбрала `RandomForestClassifier`.
- Имена параметров в Optuna **префиксуются именем модели** (`RandomForestClassifier__max_depth`), чтобы значения одной модели не попадали в другую.
- **Невалидные комбинации не роняют подбор**: ошибка внутри objective помечает trial как `PRUNED`, и Optuna идёт дальше.
- **Бюджет** соблюдается: `n_trials` или `timeout_seconds` — что наступит раньше.
- **Воспроизводимость**: `TPESampler(seed=...)` + `random_state` во всех моделях и разбиении.

---

## 5. Результаты по трём конфигам

Все запуски — `TPESampler(seed=42)`, `n_trials` по конфигу.

| Конфиг | Задача | Baseline (CV) | Победитель (CV) | Метрика на тесте |
|--------|--------|---------------|------------------|-------------------|
| churn | classification | 0.8180 | **0.8452** | 0.8141 (accuracy) |
| wine | classification | 0.9786 | **0.9717** | 1.0000 (accuracy) |
| house_prices | regression | -28300.72 | **-27879.81** | 0.9138 (R²) |

**Победители:**

- **churn:** `GradientBoostingClassifier`, `n_estimators=64`, `max_depth=5`, `numeric_impute=median`, `numeric_scale=standard`, `categorical_encode=onehot`
- **wine:** `RandomForestClassifier`, `n_estimators=100`, `min_samples_split=2`, `max_depth=3`, `numeric_impute=median`, `numeric_scale=minmax`, `categorical_encode=onehot`
- **house_prices:** `GradientBoostingRegressor`, `n_estimators=300`, `learning_rate=0.1`, `max_depth=3`, `numeric_impute=median`, `numeric_scale=standard`, `categorical_encode=onehot`

---

## 6. Отчёт по подбору

`project/reports/<config_name>.md` генерируется автоматически после каждого запуска и содержит:

- **Итог:** победитель, CV-метрика, baseline, метрика на тесте.
- **Статистику испытаний:** завершено / упало / отсечено.
- **Лидерборд топ-10:** номер trial, метрика, параметры.
- **Важность гиперпараметров:** `optuna.importance.get_param_importances`.
- **Кривую лучшего значения:** как менялся лучший результат по мере испытаний.

---

## 7. Структура проекта

```
PythonProject/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── README.md
└── project/
    ├── __init__.py
    ├── model.py                       # get_model(class_path) — динамический импорт
    ├── server.py                      # FastAPI
    ├── requirements.txt
    ├── config/
    │   ├── churn.yaml
    │   ├── wine.yaml
    │   └── house_prices.yaml
    ├── data/
    │   ├── churn.csv
    │   ├── wine.csv
    │   └── house_prices.csv
    ├── models/
    │   ├── churn/
    │   ├── wine/
    │   └── house_prices/
    ├── reports/
    │   ├── churn.md
    │   ├── wine.md
    │   └── house_prices.md
    └── src/
        ├── main.py                    # CLI: обучение + подбор + отчёт
        ├── config_loader.py
        ├── data_loader.py             # относительные пути
        ├── data_preprocessor.py       # build_preprocessor(impute, scale, encode)
        ├── model_trainer.py           # Optuna, условное пространство
        ├── report_builder.py          # генерация report.md
        └── metrics_calculator.py
```

---

## 8. API

| Метод | Путь | Смысл |
|-------|------|-------|
| GET | `/` | проверка жизни сервиса |
| GET | `/models` | список обученных `.joblib` |
| GET | `/configs` | список YAML-конфигов |
| GET | `/configs/{name}` | сам YAML-конфиг |
| GET | `/configs/{name}/models` | список моделей для конфига |
| POST | `/predict` | предсказание |