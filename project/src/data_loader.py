import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_data(config):

    project_root = "/Users/ramaldan/Desktop/PythonProject/project"
    data_path = config['dataset']['path']
    full_path = os.path.join(project_root, data_path)

    print(f"Загружаем данные из: {full_path}")
    df = pd.read_csv(full_path)
    return df


def drop_columns(df, config):
    drop_columns = config['dataset'].get('drop_columns', [])
    df = df.drop(columns=drop_columns, errors='ignore')
    return df


def separate_features_target(df, config):
    target = config['dataset']['target']
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_data(X, y, config):
    train_x, test_x, train_y, test_y = train_test_split(
        X,
        y,
        test_size=config['dataset']['split']['test_size'],
        random_state=config['dataset']['split']['random_state']
    )
    return train_x, test_x, train_y, test_y