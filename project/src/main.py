import argparse
import joblib
from pathlib import Path
from project.src.data_loader import (
    load_data,
    drop_columns,
    separate_features_target,
    split_data
)
from project.src.data_preprocessor import create_preprocessor
from project.src.model_trainer import train_model
from project.src.metrics_calculator import (
    calculate_classification_metrics,
    calculate_regression_metrics
)
from project.src.config_loader import load_config
from project.model import get_model

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    config = load_config(args.config)

    print("CONFIG:", args.config)

    df = load_data(config)

    df = drop_columns(df, config)

    X, y = separate_features_target(df, config)

    train_x, test_x, train_y, test_y = split_data(
        X,
        y,
        config
    )

    preprocessor = create_preprocessor(config)
    results = []

    for model_config in config["search_space"]["models"]:

        model_name = model_config["name"]
        model_class = model_config["class"]
        parameters = model_config["parameters"]

        print("\nОбучаем модель:", model_name)

        model = get_model(model_class)
        search = train_model(
            model,
            parameters,
            preprocessor,
            train_x,
            train_y,
            config
        )

        best_model = search.best_estimator_

        models_dir = (
                Path(__file__).resolve().parents[1]
                / "models"
                / config["name"]
        )

        models_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            best_model,
            models_dir / f"{model_name}.joblib"
        )

        prediction = search.predict(test_x)

        if config["task"] == "classification":

            probability = None

            if hasattr(search, "predict_proba"):
                probability = search.predict_proba(test_x)

            accuracy, precision, recall, f1, roc_auc = (
                calculate_classification_metrics(
                    test_y,
                    prediction,
                    probability
                )
            )

            results.append({
                "name": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "best_params": search.best_params_,
            })

        elif config["task"] == "regression":

            mae, rmse, r2 = calculate_regression_metrics(
                test_y,
                prediction
            )

            results.append({
                "name": model_name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "best_params": search.best_params_,
            })

    if results:
        print("\nРЕЗУЛЬТАТЫ:")

        for result in results:
            print(result)

    else:
        print("\nНЕТ РЕЗУЛЬТАТОВ")


if __name__ == "__main__":
    main()


# import argparse
#
# from project.src.config_loader import load_config
#
#
# def parse_arguments():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument(
#         "--config",
#         required=True,
#         help="Path to YAML configuration file"
#     )
#
#     return parser.parse_args()
#
#
# def main():
#     args = parse_arguments()
#
#     config = load_config(args.config)
#
#     print("Configuration loaded:")
#     print(config)
#
#
# if __name__ == "__main__":
#     main()
# from data_loader import load_data, drop_columns, separate_features_target, split_data
# from data_preprocessor import (
#      show_missing_values,
#      fill_numeric_na,
#      fill_categorical_na,
#      one_hot_encode,
#      align_columns,
#      check_missing_after
#  )
# from model_trainer import train_model, get_model_parameters, should_train_model
# from metrics_calculator import calculate_classification_metrics, calculate_regression_metrics
# from model import get_models
#
#
# configer = get_config_paths()
#
#
# models = get_models()
#  #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
# # #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
# for config_path in configer:
#
#      config = load_config(config_path)
#
#      print("CONFIG:", config_path)
#   #   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
#
#      df = load_data(config)
#
#
#      df = drop_columns(df, config)
#
#
#      X, y = separate_features_target(df, config)
#
#
#      train_x, test_x, train_y, test_y = split_data(X, y, config)
#
#
#      show_missing_values(train_x)
#
#
#      train_x, test_x = fill_numeric_na(train_x, test_x)
#
#
#      train_x, test_x = fill_categorical_na(train_x, test_x)
#
#
#      train_x, test_x = one_hot_encode(train_x, test_x)
#
#
#      train_x, test_x = align_columns(train_x, test_x)
#
#
#      check_missing_after(train_x)
#
#
#      results = []
#
#
#      for name, model in models.items():
#
#          if not should_train_model(config, name):
#              continue
#
#          print("\nОбучаем модель:", name)
#
#
#          parameters = get_model_parameters(config, name)
#
#
#          search = train_model(model, parameters, train_x, train_y, config)
#
#
#          prediction = search.predict(test_x)
#
#
#          if config['task'] == 'classification':
#              accuracy, precision, recall, f1 = calculate_classification_metrics(test_y, prediction)
#
#              results.append({
#                  'name': name,
#                  'accuracy': accuracy,
#                  'precision': precision,
#                  'recall': recall,
#                  'f1': f1,
#                  'best_params': search.best_params_,
#                  'model': search.best_estimator_
#              })
#
#
#          elif config['task'] == 'regression':
#              mae, rmse, r2 = calculate_regression_metrics(test_y, prediction)
#
#              results.append({
#                  'name': name,
#                  'mae': mae,
#                  'rmse': rmse,
#                  'r2': r2,
#                  'best_params': search.best_params_,
#                  'model': search.best_estimator_
#              })
#      if results:
#          print("\nРЕЗУЛЬТАТЫ:")
#          for result in results:
#              print(result)
#      else:
#          print("\nНЕТ РЕЗУЛЬТАТОВ: Ни одна модель не была обучена")