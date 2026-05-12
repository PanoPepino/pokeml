import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from pokeml.utils.utils_train import get_model, regression_metrics
from pokeml.visualisation.viz_model.residual_plot import residual_scatter


def _align_frame_to_model(X: pd.DataFrame, trained_wrapper) -> pd.DataFrame:
    """
    Align prediction dataframe to the exact feature schema used by the trained model.
    This is especially important for CatBoost native categorical handling.
    """
    X = X.copy()

    if hasattr(trained_wrapper, "model") and hasattr(trained_wrapper.model, "feature_names_"):
        expected_cols = list(trained_wrapper.model.feature_names_)

        missing = [col for col in expected_cols if col not in X.columns]
        extra = [col for col in X.columns if col not in expected_cols]

        if missing:
            raise ValueError(
                f"Missing columns for prediction: {missing}. "
                f"Model expects: {expected_cols}. "
                f"Current columns: {list(X.columns)}"
            )

        if extra:
            X = X.drop(columns=extra, errors="ignore")

        X = X[expected_cols]

    return X


def real_vs_predicted(
    input_model: str,
    input_data: dict
) -> dict:
    """
    Generate predictions, compute metrics, save residual plot and prediction dump,
    and return one metrics row for aggregation in the CLI.
    """

    the_model = joblib.load(Path(f"{input_model}.joblib"))
    model_name = get_model(input_model)

    X_train, X_val, y_train, y_val, _ = input_data[model_name]

    if isinstance(X_train, pd.DataFrame):
        X_train = _align_frame_to_model(X_train, the_model)

    if isinstance(X_val, pd.DataFrame):
        X_val = _align_frame_to_model(X_val, the_model)

    y_train_real = np.asarray(y_train).ravel()
    y_val_real = np.asarray(y_val).ravel()

    y_pred_train, uncs_train = the_model.predict_unc(X_train)
    y_pred_val, uncs_val = the_model.predict_unc(X_val)

    y_pred_train = np.asarray(y_pred_train).ravel()
    y_pred_val = np.asarray(y_pred_val).ravel()
    uncs_train = np.asarray(uncs_train).ravel()
    uncs_val = np.asarray(uncs_val).ravel()

    residuals_train = y_train_real - y_pred_train
    residuals_val = y_val_real - y_pred_val
    max_residual_val = float(np.max(np.abs(residuals_val)))

    train_metrics = regression_metrics(y_train_real, y_pred_train)
    validation_metrics = regression_metrics(y_val_real, y_pred_val)

    metrics_row = {
        "model": model_name,
        "tr_R2": float(train_metrics["R2"]),
        "tr_RMSE": float(train_metrics["RMSE"]),
        "tr_MAE": float(train_metrics["MAE"]),
        "val_R2": float(validation_metrics["R2"]),
        "val_RMSE": float(validation_metrics["RMSE"]),
        "val_MAE": float(validation_metrics["MAE"]),
        "overfit_R2": float(train_metrics["R2"] - validation_metrics["R2"]),
        "overfit_RMSE": float(train_metrics["RMSE"] - validation_metrics["RMSE"]),
        "max_res": max_residual_val,
    }

    p = Path(input_model)
    last = p.name

    residual_scatter(last, y_pred_val, y_val_real, uncs_val)

    out_dir = Path("artifacts/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_df = pd.DataFrame({
        "model": model_name,
        "y_true": y_val_real,
        "y_pred": y_pred_val,
        "unc": uncs_val,
        "residual": residuals_val,
        "split": "validation",
    })
    preds_df.to_csv(out_dir / f"preds_{last}.csv", index=False)

    return metrics_row
