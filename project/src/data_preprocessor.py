import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def create_preprocessor(config):
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

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ])

    return preprocessor