import yaml

def load_config(config_path):

    full_path = f"/Users/ramaldan/Desktop/PythonProject/project/{config_path}"
    with open(full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_config_paths():
    return [
        'config/churn.yaml',
        'config/house_prices.yaml',
        'config/wine.yaml'
    ]