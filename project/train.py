import pandas as pd
import yaml

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from model import get_models
# ВСЕ CONFIG
configer = [
    'configs/churn.yaml',
    'configs/house_prices.yaml',
    'configs/wine.yaml'
]
# ВСЕ МОДЕЛИ
models = get_models()
# ОБУЧАЕМ КАЖДЫЙ DATASET
for config_path in configer:
    # YAML
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("\n===================================")
    print("CONFIG:", config_path)
    print("===================================")
    # ЗАГРУЗКА DATASET
    df = pd.read_csv(
        config['dataset']['path']
    )

    target = config['dataset']['target']
    # УДАЛЯЕМ НЕНУЖНЫЕ СТОЛБЦЫ
    drop_columns = config['features'].get(
        'drop_columns',
        []
    )

    df = df.drop(
        columns=drop_columns,
        errors='ignore'
    )
    # X / Y
    X = df.drop(
        columns=[target]
    )

    y = df[target]
    # SPLIT
    train_x, test_x, train_y, test_y = train_test_split(
        X,
        y,
        test_size=config['dataset']['split']['test_size'],
        random_state=config['dataset']['split']['random_state']
    )
    # ПРЕДОБРАБОТКА
    print("\nПропуски до обработки:")

    print(
        train_x.isnull().sum()
        .loc[lambda x: x > 0]
    )
    # ЧИСЛОВЫЕ СТОЛБЦЫ
    numeric_columns = train_x.select_dtypes(
        include=['number']
    ).columns
    # КАТЕГОРИАЛЬНЫЕ СТОЛБЦЫ

    categorical_columns = train_x.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns
    # ЗАПОЛНЯЕМ NUMERIC NaN

    for column in numeric_columns:

        median = train_x[column].median()

        train_x[column] = train_x[column].fillna(
            median
        )

        test_x[column] = test_x[column].fillna(
            median
        )
    # ЗАПОЛНЯЕМ CATEGORICAL NaN
    for column in categorical_columns:

        mode = train_x[column].mode()

        if len(mode) > 0:

            value = mode[0]

            train_x[column] = train_x[column].fillna(
                value
            )

            test_x[column] = test_x[column].fillna(
                value
            )
    # ONE-HOT ENCODING

    train_x = pd.get_dummies(
        train_x,
        drop_first=True
    )

    test_x = pd.get_dummies(
        test_x,
        drop_first=True
    )
    # ДЕЛАЕМ ОДИНАКОВЫЕ СТОЛБЦЫ
    train_x, test_x = train_x.align(
        test_x,
        join='left',
        axis=1,
        fill_value=0
    )
    # ПРОВЕРЯЕМ NaN
    print("\nПропуски после обработки:")

    print(
        train_x.isnull().sum()
        .loc[lambda x: x > 0]
    )
    # RESULTS
    results = []
    # МОДЕЛИ
    for name, model in models.items():
        # БЕРЕМ ТОЛЬКО МОДЕЛИ ИЗ ЭТОГО YAML
        if name not in config['search_space']['models']:
            continue


        print(
            "\nОбучаем модель:",
            name
        )
        # PARAMETERS
        parameters = (
            config['search_space']
            ['models']
            [name]
            ['parameters']
        )
        # GRID SEARCH
        search = GridSearchCV(
            estimator=model,
            param_grid=parameters,
            cv=config['search']['cv'],
            scoring=config['search']['scoring']
        )
        # TRAIN
        search.fit(
            train_x,
            train_y
        )
        # PREDICTION
        prediction = search.predict(
            test_x
        )
        # CLASSIFICATION
        if config['task'] == 'classification':

            accuracy = accuracy_score(
                test_y,
                prediction
            )

            precision = precision_score(
                test_y,
                prediction,
                average='weighted',
                zero_division=0
            )

            recall = recall_score(
                test_y,
                prediction,
                average='weighted',
                zero_division=0
            )

            f1 = f1_score(
                test_y,
                prediction,
                average='weighted',
                zero_division=0
            )


            results.append({
                'name': name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'best_params': search.best_params_,
                'model': search.best_estimator_
            })

        # REGRESSION
        elif config['task'] == 'regression':

            mae = mean_absolute_error(
                test_y,
                prediction
            )

            rmse = mean_squared_error(
                test_y,
                prediction
            ) ** 0.5

            r2 = r2_score(
                test_y,
                prediction
            )


            results.append({
                'name': name,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'best_params': search.best_params_,
                'model': search.best_estimator_
            })

    # RESULTS
    print("\nРЕЗУЛЬТАТЫ:")

    for result in results:

        print(result)