from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression


def get_model(name):
    models = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }

    if name not in models:
        raise ValueError(f"Неизвестная модель: {name}")

    return models[name]