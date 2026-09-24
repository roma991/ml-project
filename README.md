Universal ML Training System

Универсальная система для обучения и тестирования моделей машинного обучения с конфигурацией через YAML-файлы.

Проект позволяет использовать один ML pipeline для разных датасетов, задач классификации и регрессии, моделей и вариантов preprocessing без изменения основного кода обучения.

Возможности

* YAML-конфигурация ML-пайплайна
* Поддержка классификации и регрессии
* Несколько моделей в одном конфиге
* GridSearchCV для подбора гиперпараметров
* Подбор параметров preprocessing через GridSearchCV
* Обработка числовых и категориальных признаков
* Imputation пропущенных значений
* StandardScaler и MinMaxScaler
* OneHotEncoder и OrdinalEncoder
* Расчёт метрик для classification и regression
* Сохранение обученных моделей в формате .joblib
* REST API на FastAPI
* Возможность добавлять новые датасеты и модели через YAML

Архитектура

PythonProject/
│
├── project/
│   ├── config/
│   │   ├── churn.yaml
│   │   ├── house_prices.yaml
│   │   ├── project.yaml
│   │   └── wine.yaml
│   │
│   ├── data/
│   │   ├── churn.csv
│   │   ├── house_prices.csv
│   │   └── wine.csv
│   │
│   ├── models/
│   │   ├── churn/
│   │   ├── house_prices/
│   │   └── wine/
│   │
│   ├── src/
│   │   ├── config_loader.py
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   ├── main.py
│   │   ├── metrics_calculator.py
│   │   └── model_trainer.py
│   │
│   ├── model.py
│   ├── predict.py
│   └── server.py
│
├── .gitignore
└── README.md

Как работает pipeline

Основной pipeline построен следующим образом:

YAML configuration
        ↓
Load dataset
        ↓
Drop unnecessary columns
        ↓
Separate features and target
        ↓
Train / test split
        ↓
Preprocessing
        ↓
GridSearchCV
        ↓
Model training
        ↓
Evaluation
        ↓
Save best model
        ↓
FastAPI

Основная логика обучения находится в project/src/main.py.

Конфигурация загружается из YAML:

project/config/

Поэтому Python-код обучения не нужно изменять при переходе между разными датасетами.

Конфигурация

Каждый датасет имеет собственный YAML-файл.

Например:

project/config/churn.yaml

Конфигурация определяет:

* тип задачи;
* путь к датасету;
* target;
* признаки;
* удаляемые столбцы;
* параметры train/test split;
* preprocessing;
* модели;
* диапазоны гиперпараметров;
* метрики;
* параметры cross-validation.

Поддерживаемые задачи

Classification

В проекте используются:

* RandomForestClassifier
* GradientBoostingClassifier

Для классификации рассчитываются:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC

Regression

Для регрессии используется:

* GradientBoostingRegressor

Рассчитываются:

* MAE
* RMSE
* R²

Используемые датасеты

Churn

Задача бинарной классификации оттока клиентов.

Модели:

* RandomForestClassifier
* GradientBoostingClassifier

Основная метрика:

ROC-AUC

House Prices

Задача регрессии стоимости недвижимости.

Модель:

* GradientBoostingRegressor

Рассчитываются:

MAE
RMSE
R²

Wine

Задача многоклассовой классификации.

Модели:

* RandomForestClassifier
* GradientBoostingClassifier

Рассчитываются:

Accuracy
Precision
Recall
F1
ROC-AUC

Запуск обучения

Из директории PythonProject:

Churn

python -m project.src.main --config project/config/churn.yaml

House Prices

python -m project.src.main --config project/config/house_prices.yaml

Wine

python -m project.src.main --config project/config/wine.yaml

После обучения модели сохраняются в:

project/models/{dataset}/

Например:

project/models/churn/RandomForestClassifier.joblib

GridSearchCV

Для обучения используется GridSearchCV.

Поиск может включать не только гиперпараметры модели, но и параметры preprocessing.

Например:

Model parameters
        +
Imputer strategy
        +
Scaler
        +
Categorical encoder
        ↓
GridSearchCV

Таким образом, система автоматически выбирает комбинацию preprocessing и параметров модели с лучшим результатом на cross-validation.

FastAPI

Обученные модели доступны через REST API.

После запуска сервера документация Swagger доступна по адресу:

http://127.0.0.1:8002/docs

API принимает:

* название конфигурации;
* название модели;
* значения признаков.

Пример структуры запроса:

{
    "config": "churn",
    "model": "RandomForestClassifier",
    "features": {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5
    }
}

FastAPI загружает соответствующую .joblib модель и выполняет prediction.

Для классификации API также может возвращать вероятности классов.

Добавление нового датасета

Чтобы добавить новый датасет:

1. Добавить CSV в:

project/data/

2. Создать YAML-конфигурацию:

project/config/new_dataset.yaml

3. Описать в YAML:

task:
name:
dataset:
features:
metric:
search:
search_space:

4. Запустить:

python -m project.src.main --config project/config/new_dataset.yaml

Основной Python pipeline при этом изменять не требуется.

Основной стек

* Python
* pandas
* scikit-learn
* PyYAML
* joblib
* FastAPI
* Uvicorn

Цель проекта

Проект демонстрирует построение переиспользуемого ML pipeline, в котором конфигурация эксперимента отделена от Python-кода.

Основная идея:

Configuration → Pipeline → Training → Evaluation → Model → API