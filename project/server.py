from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
CONFIGS_DIR = BASE_DIR / "config"


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "ML API is running"
    }


@app.get("/models")
def get_models():
    models = [
        path.stem
        for path in MODELS_DIR.glob("*.joblib")
    ]

    return {
        "models": models
    }


@app.get("/configs")
def get_configs():
    configs = [
        path.stem
        for path in CONFIGS_DIR.glob("*.yaml")
    ]

    return {
        "configs": configs
    }


def load_config(config_name: str):
    config_path = CONFIGS_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config '{config_name}' not found"
        )

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


@app.get("/configs/{config_name}")
def get_config(config_name: str):
    config_path = CONFIGS_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        return {
            "error": f"Config '{config_name}' not found"
        }

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def validate_features(config, features):

    errors = []

    expected_features = []

    for feature in config["features"]["numeric"]:
        expected_features.append(feature["name"])

    for feature in config["features"]["categorial"]:
        expected_features.append(feature["name"])

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features
    ]

    extra_features = [
        feature
        for feature in features
        if feature not in expected_features
    ]

    if missing_features:
        errors.append({
            "type": "missing_features",
            "features": missing_features
        })

    if extra_features:
        errors.append({
            "type": "extra_features",
            "features": extra_features
        })

    for feature in config["features"]["categorial"]:
        name = feature["name"]

        if name in features and "values" in feature:
            allowed_values = feature["values"]

            if features[name] not in allowed_values:
                errors.append({
                    "type": "invalid_value",
                    "feature": name,
                    "value": features[name],
                    "allowed_values": allowed_values
                })

    return errors
@app.get("/configs/{config_name}/models")
def get_config_models(config_name: str):

    config = load_config(config_name)


    models = config["search_space"]["models"]

    available_models = []

    for model_name in models:
        model_path = MODELS_DIR / config_name /f"{model_name}.joblib"

        if model_path.exists():
            available_models.append(model_name)

    return {
        "config": config_name,
        "task": config["task"],
        "models": available_models
    }

def validate_features(config, features):

    errors = []

    target = config["dataset"]["target"]
    drop_columns = config["dataset"]["drop_columns"]

    expected_features = []

    for feature in config["features"]["numeric"]:
        expected_features.append(feature["name"])

    for feature in config["features"]["categorial"]:
        expected_features.append(feature["name"])

    # Убираем target
    expected_features.remove(target)

    # Убираем колонки, которые должны быть удалены
    expected_features = [
        feature
        for feature in expected_features
        if feature not in drop_columns
    ]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in features
    ]

    extra_features = [
        feature
        for feature in features
        if feature not in expected_features
    ]

    if missing_features:
        errors.append({
            "type": "missing_features",
            "features": missing_features
        })

    if extra_features:
        errors.append({
            "type": "extra_features",
            "features": extra_features
        })

    for feature in config["features"]["categorial"]:

        name = feature["name"]

        if name not in expected_features:
            continue

        if name in features and "values" in feature:

            allowed_values = feature["values"]

            if features[name] not in allowed_values:
                errors.append({
                    "type": "invalid_value",
                    "feature": name,
                    "value": features[name],
                    "allowed_values": allowed_values
                })

    return errors
@app.post("/predict")
def predict(data: dict):

    config_name = data["config"]
    model_name = data["model"]
    features = data["features"]

    # Загружаем YAML
    config = load_config(config_name)

    # Проверяем признаки
    errors = validate_features(config, features)

    if errors:
        return {
            "error": "Invalid features",
            "details": errors
        }

    # Проверяем модель
    allowed_models = config["search_space"]["models"]

    if model_name not in allowed_models:
        return {
            "error": f"Model '{model_name}' is not allowed for config '{config_name}'",
            "allowed_models": list(allowed_models.keys())
        }

    # Проверяем наличие файла модели
    model_path = MODELS_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        return {
            "error": f"Model '{model_name}.joblib' not found"
        }

    # Загружаем модель
    model = joblib.load(model_path)

    # Создаём DataFrame
    df = pd.DataFrame([features])

    # Prediction
    prediction = model.predict(df)[0]

    # Результат
    result = {
        "config": config_name,
        "task": config["task"],
        "model": model_name,
        "prediction": prediction
    }

    # Для классификации добавляем вероятность
    if config["task"] == "classification":

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(df)[0]

            result["probabilities"] = {
                str(class_name): float(probability)
                for class_name, probability
                in zip(model.classes_, probabilities)
            }

    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002
    )