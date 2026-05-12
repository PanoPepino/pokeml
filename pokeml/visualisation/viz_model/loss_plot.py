import matplotlib.pyplot as plt

from pokeml.visualisation.constants_plot import *


TRAIN_COLOR = MAIN_COLOR
VALID_COLOR = SECOND_COLOR      # replace with your chosen validation color


def _pick_first_existing(dct, candidates):
    for key in candidates:
        if key in dct:
            return key
    return None


def _pick_metric(dct, candidates):
    for key in candidates:
        if key in dct:
            return key
    return None


def loss_plot(name_model, evals, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    # CatBoost-style eval dict
    if "learn" in evals:
        train_key = "learn"
        valid_key = _pick_first_existing(evals, ["validation_1", "validation", "validation_0"])

        if valid_key is None:
            raise KeyError(
                f"No CatBoost validation key found for {name_model}. "
                f"Available keys: {list(evals.keys())}"
            )

        train_metric = _pick_metric(
            evals[train_key],
            ["RMSEWithUncertainty", "RMSE", "MultiClass", "Accuracy"]
        )
        valid_metric = _pick_metric(
            evals[valid_key],
            ["RMSEWithUncertainty", "RMSE", "MultiClass", "Accuracy"]
        )

        if train_metric is None or valid_metric is None:
            raise KeyError(
                f"Could not find CatBoost metric for {name_model}. "
                f"Train metrics: {list(evals[train_key].keys())}, "
                f"Valid metrics: {list(evals[valid_key].keys())}"
            )

        learn = evals[train_key][train_metric]
        validation = evals[valid_key][valid_metric]
        metric_name = valid_metric

    # LightGBM-style eval dict
    else:
        train_key = _pick_first_existing(evals, ["training", "valid_0"])
        valid_key = _pick_first_existing(evals, ["valid_1", "valid_0"])

        if train_key is None or valid_key is None:
            raise KeyError(
                f"No LightGBM train/valid keys found for {name_model}. "
                f"Available keys: {list(evals.keys())}"
            )

        train_metric = _pick_metric(evals[train_key], ["quantile", "l2", "rmse"])
        valid_metric = _pick_metric(evals[valid_key], ["quantile", "l2", "rmse"])

        if train_metric is None or valid_metric is None:
            raise KeyError(
                f"Could not find LightGBM metric for {name_model}. "
                f"Train metrics: {list(evals[train_key].keys())}, "
                f"Valid metrics: {list(evals[valid_key].keys())}"
            )

        learn = evals[train_key][train_metric]
        validation = evals[valid_key][valid_metric]
        metric_name = valid_metric

    ax.plot(
        learn,
        label="train",
        linewidth=2,
        alpha=ALPHA,
        color=TRAIN_COLOR
    )
    ax.plot(
        validation,
        label="validation",
        linewidth=2,
        alpha=ALPHA,
        color=VALID_COLOR
    )

    ax.set_title(name_model)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(alpha=0.25)

    return ax
