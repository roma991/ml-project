from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def train_model(
    model,
    parameters,
    preprocessor,
    train_x,
    train_y,
    config
):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline_parameters = {}

    # Параметры модели
    for parameter, values in parameters.items():
        pipeline_parameters[f"model__{parameter}"] = values

    # Preprocessing из YAML
    preprocessing_config = config["search_space"]["preprocessing"]

    # Заполнение пропусков
    pipeline_parameters[
        "preprocessor__numeric__imputer__strategy"
    ] = preprocessing_config["numeric_impute"]["values"]

    # Масштабирование
    scalers = []

    for scaler in preprocessing_config["numeric_scale"]["values"]:

        if scaler == "standard":
            scalers.append(StandardScaler())

        elif scaler == "minmax":
            scalers.append(MinMaxScaler())

    pipeline_parameters[
        "preprocessor__numeric__scaler"
    ] = scalers

    # Кодирование категориальных признаков
    encoders = []

    for encoder in preprocessing_config["categorical_encode"]["values"]:

        if encoder == "onehot":
            encoders.append(
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        elif encoder == "ordinal":
            encoders.append(
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            )

    pipeline_parameters[
        "preprocessor__categorical__encoder"
    ] = encoders

    # GridSearchCV
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_parameters,
        cv=config["search"]["cv"],
        scoring=config["search"]["scoring"]
    )

    search.fit(train_x, train_y)

    return search