import json
import joblib


from pathlib import Path
from pokeml.models.trainers import Cat_Trainer, LGBM_Trainer
from pokeml.visualisation.viz_model.training_curves_plot import training_curves_plot


def train(prep_data,
          params: dict,
          output_name,
          classifier=None,  #  Added Edu's idea into the pipeline
          fe_state=None):
    """
    Train models and save one combined figure with 3 loss-curve subplots
    (one subplot per model).
    """

    if not hasattr(prep_data, "items"):
        raise TypeError(
            "prep_data must be a dict like {'cat_ordinal': (...), ...}. If you have (data, fe_state), pass only data.")

    trained_models = {}
    evals = []  # For plots
    summary = []  # Recommended summary
    results = []  # Similarly as tuning, display more info

    # Defining path for out
    p = Path(output_name)
    out_dir = p.parent
    last = p.name

    out_dir.mkdir(parents=True, exist_ok=True)

    if fe_state is not None:
        fe_path = out_dir / f"{last}_fe_state.joblib"
        joblib.dump(fe_state, fe_path, compress=3)

    for name, (X_tr_p, X_te_p, y_tr, y_te, cats) in prep_data.items():

        # --------------------------------------------
        # Adding Edu's classifier. Called step 1
        #  Will add prebands and the probabilities of belonging to each band
        # Careful! preband is a categorical feat.
        # --------------------------------------------

        if classifier is not None:
            X_tr_p = classifier.enrich(X_tr_p)
            X_te_p = classifier.enrich(X_te_p)

            if "pred_band" not in cats:
                cats = cats + ["pred_band"]

        feature_names = list(X_tr_p.columns)
        joblib.dump(feature_names, f"{output_name}_{name}_features.joblib")

        # --------------------------------------------
        # Here goes the previous logic. Called step 2
        # --------------------------------------------

        if name in ("cat_ordinal", "cat_native"):
            trainer = Cat_Trainer(cat_features=cats, **params[name])
        else:
            trainer = LGBM_Trainer(**params["light_gbm"])

        eval_full = [(X_tr_p, y_tr), (X_te_p, y_te)]
        pipeline = trainer.fit(X_tr_p, y_tr, eval_set=eval_full)
        trained_models[name] = pipeline

        # Saving each model in joblib + information in summary
        model_path = out_dir / f"{last}_{name}.joblib"
        joblib.dump(pipeline, model_path, compress=3)

        summary.append({
            "model_name": name,
            "model_path": str(model_path),
            "n_train_rows": len(X_tr_p),
            "n_test_rows": len(X_te_p)
        })

        # Saving evaluations for easier plotting of train/validation curves
        test_evals = trainer.get_evals()
        evals.append({
            "model_name": f"{last}_{name}",
            "evals": test_evals
        })

    # One single figure with 3 subplots, one for each model

    training_curves_plot(last, evals)

    summary_path = out_dir / f"{last}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return trained_models
