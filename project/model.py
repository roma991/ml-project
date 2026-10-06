import importlib

def get_model(class_path, parameters=None):
    parameters = parameters or {}

    module_name, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_name)

    if not hasattr(module, class_name):
        raise ValueError(
            f"Класс модели '{class_name}' не найден "
            f"в модуле '{module_name}'"
        )

    model_class = getattr(module, class_name)

    return model_class(**parameters)