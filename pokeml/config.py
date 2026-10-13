"""Local experiment configuration and tuning-grid expansion."""
from pathlib import Path
import numpy as np
import json

def expand_arange(param_def):
    """
    Convert {'start': 0.1, 'stop': 1.0, 'step': 0.1} → [0.1, 0.2, ..., 0.9]
    """

    if isinstance(param_def, dict):
        return np.arange(param_def["start"], param_def["stop"], param_def["step"]).tolist()
    return param_def

def load_tuning_grid(config_path: Path):
    """
    Load compact JSON and expand to full GridSearchCV param_grid.
    """

    with open(config_path) as f:
        config = json.load(f)

    # Expand common params
    common = {k: expand_arange(v) for k, v in config["common_params"].items()}

    # Merge into model-specific grids
    params_grid = {}
    for model_name, model_params in config["params_grid"].items():
        full_params = {**common, **{k: expand_arange(v) for k, v in model_params.items()}}
        params_grid[model_name] = full_params

    return params_grid



