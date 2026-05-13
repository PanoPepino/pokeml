import pandas as pd
import json
import numpy as np

from pathlib import Path
from rich.console import Console
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from pokeml.models.trainers import Cat_Trainer, LGBM_Trainer
from pokeml.utils.utils_train import load_tuning_grid
from pokeml.utils.utils_commands import CliUI


console = Console()

ui = CliUI()


def tuning(prep_data,
           my_grid,
           search_iter,
           output_name,
           classifier=None):
    """
    Search the best possible parameters to fit the model for the best predictions.

    Args:
        prep_data (DataFrame): The data already encoded, imputed and properly massaged.
        my_grid (str): Path to the JSON grid of parameters to search.
        search_iter (int): The amount of times the process repeats.
        output_name (str): The name of the output file (w/o extension).
        classifier (BandClassifier, optional):
            A fitted BandClassifier instance. When provided, each split is enriched
            with `pred_band` + `proba_*` columns before tuning — mirroring the
            behaviour of train() so that tuning and training share the same
            feature space.

    Returns:
        Dict: Best parameters per model saved as .json; CV results saved as .csv.
    """

    # Load Grid of parameters
    some_params = load_tuning_grid(my_grid)

    # Create empty rows to be appended with info.
    results = []
    future_fit = {}

    for name, (X_tr_p, X_te_p, y_tr, y_te, cats) in prep_data.items():

        # ----------------------------------------------------------------
        # Stage 1 enrichment: apply BandClassifier if provided.
        # This ensures tuning runs on the same enriched feature space
        # that train() uses, so the found parameters are truly compatible.
        # ----------------------------------------------------------------
        if classifier is not None:
            X_tr_p = classifier.enrich(X_tr_p)
            X_te_p = classifier.enrich(X_te_p)
            if "pred_band" not in cats:
                cats = cats + ["pred_band"]

        if name in ("cat_ordinal", "cat_native"):
            trainer = Cat_Trainer(cat_features=cats)
            cv_kwargs = some_params['catboost']
            fit_kwargs = {'eval_set': (X_te_p, y_te)}

        else:
            trainer = LGBM_Trainer()
            cv_kwargs = some_params['light_gbm']
            fit_kwargs = {'eval_set': [(X_te_p, y_te)]}

        search = RandomizedSearchCV(trainer,
                                    cv_kwargs,
                                    n_iter=search_iter,
                                    cv=search_iter+1,
                                    scoring='neg_mean_absolute_error',
                                    random_state=1,
                                    verbose=0)
        search.fit(X_tr_p, y_tr)

        best = search.best_estimator_
        best.fit(X_tr_p, y_tr, **fit_kwargs)
        y_pred = best.predict(X_te_p)
        r2_preds = r2_score(y_te, y_pred)
        rmse_preds = np.sqrt(mean_squared_error(y_te, y_pred))
        max_res = np.max(np.abs(y_te - y_pred))

        feature_names = X_tr_p.columns.tolist()
        top_features, top_importances = best.get_top_features(feature_names, k=5)

        results.append({
            'model': name,
            'iterations': search_iter,
            'tuning_MAE': -search.best_score_,  # Positive MAE
            'test_R2': r2_preds,
            'test_RMSE': rmse_preds,
            'test_MAE': mean_absolute_error(y_te, y_pred),
            'max_residual': max_res,
            'top_features': json.dumps(top_features),
            'top_feature_weights': json.dumps(top_importances)
        })

        future_fit.update({
            name: search.best_params_
        })

    # Defining outputs
    p = Path(output_name)
    parent = str(p.parent)
    last = p.name

    out_dir = Path(parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_data = pd.DataFrame(results)
    cv_data.to_csv(f"{parent}/{last}_cv.csv", index=False)

    best_params_path = Path(f"{output_name}_bp.json")  # bp = best_params
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(future_fit, f, indent=2)
