from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)

    if path.exists():
        full_path = path
    else:
        project_dir = Path(__file__).resolve().parent.parent
        full_path = project_dir / path

    with open(full_path, "r") as file:
        return yaml.safe_load(file)