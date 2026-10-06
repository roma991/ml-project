import argparse
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score

from project.src.data_loader import (
    load_data,
    drop_columns,
    separate_features_target,
    split_data,
)
from project.src.data_preprocessor import build_preprocessor
from project.src.model_trainer import train_model
from project.src.report_builder import generate_report
from project.src.config_loader import load_config
from project.model import get_model


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_config(args.config)
    print("CONFIG:", args.config)

    df = load_data(config)
    df = drop_columns(df, config)
    X, y = separate_features_target(df, config)
    train_x, test_x, train_y, test_y = split_data(X, y, config)

    def preprocessor_factory(impute, scale, encode):
        return build_preprocessor(config, impute, scale, encode)

    # --- 1. БАЗА (default params, первый кандидат) ---
    first_model_cfg = config["search_space"]["models"][0]
    default_model = get_model(first_model_cfg["class"])
    default_preprocessor = build_preprocessor(
        config, "median", "standard", "onehot"
    )
    default_pipeline = Pipeline([
        ("preprocessor", default_preprocessor),
        ("model", default_model),
    ])
    default_pipeline.fit(train_x, train_y)

    baseline_metric = float(
        cross_val_score(
            default_pipeline,
            train_x,
            train_y,
            cv=config["tuning"]["cv"]["folds"],
            scoring=config["search"]["scoring"],
        ).mean()
    )

    print(
        f"\nBaseline ({first_model_cfg['name']}, default params): "
        f"{baseline_metric:.4f}"
    )

    # --- 2. OPTUNA ---
    print("\nЗапускаем Optuna...")
    study = train_model(config, preprocessor_factory, train_x, train_y)

    print(f"Лучшее значение CV: {study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")

    # --- 3. Восстанавливаем победителя ---
    best = study.best_params
    best_model_name = best["model"]
    prefix = f"{best_model_name}__"

    best_model_cfg = next(
        m for m in config["search_space"]["models"]
        if m["name"] == best_model_name
    )

    best_params = {
        k.replace(prefix, ""): v
        for k, v in best.items()
        if k.startswith(prefix)
    }

    best_model = get_model(best_model_cfg["class"], best_params)
    best_preprocessor = build_preprocessor(
        config,
        best["numeric_impute"],
        best["numeric_scale"],
        best["categorical_encode"],
    )

    best_pipeline = Pipeline([
        ("preprocessor", best_preprocessor),
        ("model", best_model),
    ])
    best_pipeline.fit(train_x, train_y)

    # --- 4. Метрика на тесте (один раз) ---
    test_prediction = best_pipeline.predict(test_x)

    if config["task"] in (
        "classification",
        "binary_classification",
        "multiclass_classification",
    ):
        test_metric = float(accuracy_score(test_y, test_prediction))
    else:
        test_metric = float(r2_score(test_y, test_prediction))

    print(f"Метрика на тесте: {test_metric:.4f}")

    # --- 5. Сохраняем артефакт с метаданными ---
    models_dir = Path(__file__).resolve().parents[1] / "models" / config["name"]
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": best_pipeline,
        "metadata": {
            "config_name": config["name"],
            "config_version": config.get("version"),
            "trained_at": datetime.now().isoformat(),
            "winner_model": best_model_name,
            "winner_params": best_params,
            "preprocessing": {
                "numeric_impute": best["numeric_impute"],
                "numeric_scale": best["numeric_scale"],
                "categorical_encode": best["categorical_encode"],
            },
            "cv_metric": study.best_value,
            "test_metric": test_metric,
            "baseline_metric": baseline_metric,
        },
    }

    artifact_path = models_dir / f"{best_model_name}.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"Артефакт сохранён: {artifact_path}")

    # --- 6. Отчёт ---
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    generate_report(
        study=study,
        config=config,
        test_metric=test_metric,
        baseline_metric=baseline_metric,
        output_path=reports_dir / f"{config['name']}.md",
    )


if __name__ == "__main__":
    main()