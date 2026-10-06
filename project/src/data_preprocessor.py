from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_preprocessor(config, numeric_impute, numeric_scale, categorical_encode):
    """Фабрика препроцессора — параметры приходят из Optuna trial."""

    numeric_columns = [
        feature["name"]
        for feature in config["features"]["numeric"]
        if feature["name"] not in config["dataset"].get("drop_columns", [])
        and feature["name"] != config["dataset"]["target"]
    ]

    categorical_columns = [
        feature["name"]
        for feature in config["features"]["categorial"]
        if feature["name"] != config["dataset"]["target"]
        and feature["name"] not in config["dataset"].get("drop_columns", [])
    ]

    if numeric_scale == "standard":
        scaler = StandardScaler()
    elif numeric_scale == "minmax":
        scaler = MinMaxScaler()
    elif numeric_scale == "none":
        scaler = "passthrough"
    else:
        raise ValueError(f"Неизвестный scaler: {numeric_scale}")

    if categorical_encode == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif categorical_encode == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"Неизвестный encoder: {categorical_encode}")

    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_impute))]
    if scaler != "passthrough":
        numeric_steps.append(("scaler", scaler))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])