from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data(config):
    data_path = PROJECT_ROOT / config["dataset"]["path"]

    if not data_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {data_path}")

    print(f"Загружаем данные из: {data_path}")
    return pd.read_csv(data_path)


def drop_columns(df, config):
    drop_columns = config["dataset"].get("drop_columns", [])
    return df.drop(columns=drop_columns, errors="ignore")


def separate_features_target(df, config):
    target = config["dataset"]["target"]
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_data(X, y, config):
    train_x, test_x, train_y, test_y = train_test_split(
        X,
        y,
        test_size=config["dataset"]["split"]["test_size"],
        random_state=config["dataset"]["split"]["random_state"],
    )
    return train_x, test_x, train_y, test_y