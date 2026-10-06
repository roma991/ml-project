import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from project.model import get_model


def _suggest_param(trial, model_name, param_name, spec):
    """Уникальное имя параметра — префикс с именем модели.

    Иначе Optuna считает max_depth одним параметром для всех моделей,
    и значения из одной модели попадают в другую.
    """
    key = f"{model_name}__{param_name}"

    if isinstance(spec, list):
        return trial.suggest_categorical(key, spec)

    if isinstance(spec, dict):
        spec_type = spec.get("type")

        if spec_type == "int":
            return trial.suggest_int(
                key,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
            )

        if spec_type == "float":
            return trial.suggest_float(
                key,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )

        if spec_type == "categorical":
            return trial.suggest_categorical(key, spec["values"])

    raise ValueError(f"Неизвестный формат параметра '{param_name}': {spec}")


def train_model(config, preprocessor_factory, train_x, train_y):
    """Запускает Optuna-подбор и возвращает study."""

    search_models = config["search_space"]["models"]
    preprocessing = config["search_space"]["preprocessing"]

    scoring = config["search"]["scoring"]
    direction = config["tuning"]["direction"]
    cv_folds = config["tuning"]["cv"]["folds"]
    seed = config["tuning"]["seed"]
    n_trials = config["tuning"]["budget"]["n_trials"]
    timeout = config["tuning"]["budget"]["timeout_seconds"]

    model_names = [m["name"] for m in search_models]
    model_by_name = {m["name"]: m for m in search_models}

    def objective(trial):
        impute = trial.suggest_categorical(
            "numeric_impute", preprocessing["numeric_impute"]["values"]
        )
        scale = trial.suggest_categorical(
            "numeric_scale", preprocessing["numeric_scale"]["values"]
        )
        encode = trial.suggest_categorical(
            "categorical_encode", preprocessing["categorical_encode"]["values"]
        )

        model_name = trial.suggest_categorical("model", model_names)
        model_config = model_by_name[model_name]

        params = {}
        for param_name, spec in model_config["parameters"].items():
            params[param_name] = _suggest_param(
                trial, model_name, param_name, spec
            )

        try:
            model = get_model(model_config["class"], params)
            preprocessor = preprocessor_factory(impute, scale, encode)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            scores = cross_val_score(
                pipeline,
                train_x,
                train_y,
                cv=cv_folds,
                scoring=scoring,
            )
            return float(scores.mean())

        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            raise optuna.TrialPruned()

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
    )

    return study