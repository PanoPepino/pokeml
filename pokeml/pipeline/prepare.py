# pokeml/pipeline/prepare.py
# Responsible for: orchestrating feature engineering, cleaning,
# splitting, and packaging data for all three model variants.
# The three prep_* encoding functions live in features/preprocess.py.
# This layer calls them — it does not define them.

import pandas as pd
from sklearn.model_selection import train_test_split

from pokeml.data.load import load_data
from pokeml.data.catfill import cat_fill
from pokeml.features.preprocess import (
    prep_catboost_ordinal,
    prep_catboost_native,
    prep_lightgbm,
)


def prepare_data_train(df,
                       tsize=0.3,
                       feat_eng_steps=None):
    """
    Full training-data pipeline:
      1. Load raw data.
      2. Apply feature engineering steps (compute + apply).
      3. Clean and cast with cat_fill.
      4. Train/test split.
      5. Run all three prep_* encoders.
      6. Return a dict keyed by model variant + the fe_state.

    Args:
        df:              Path string or DataFrame.
        tsize:           Test fraction (default 0.3).
        feat_eng_steps:  List of (step_name, compute_fn, apply_fn) tuples.

    Returns:
        data (dict):  Keys = ["cat_ordinal", "cat_native", "light_gbm"].
                      Values = (X_tr, X_te, y_tr, y_te, cats).
        fe_state (dict): Computed state from each feature engineering step,
                         needed to replay the same transforms at predict time.
    """
    raw_df = load_data(df)

    fe_state = {}
    feat_eng_steps = feat_eng_steps or []

    all_new_cat_feats = []
    all_new_maps = {}

    for step_name, compute_func, apply_func in feat_eng_steps:
        step_state = compute_func(raw_df)
        fe_state[step_name] = step_state
        raw_df = apply_func(raw_df, step_state)

        step_cat_feats = step_state.get("new_cat_feats", [])
        if step_cat_feats:
            all_new_cat_feats.extend(step_cat_feats)

        step_maps = step_state.get("new_maps", {})
        if step_maps:
            all_new_maps.update(step_maps)

    df_to_split = cat_fill(raw_df)
    Xy_tr, Xy_te = train_test_split(df_to_split, test_size=tsize, random_state=1)

    prep = {
        "cat_ordinal": prep_catboost_ordinal,
        "cat_native":  prep_catboost_native,
        "light_gbm":   prep_lightgbm,
    }

    data = {}
    for name, func in prep.items():
        if name == "cat_ordinal":
            tr_out, cats = func(Xy_tr,
                                new_cat_feats=all_new_cat_feats,
                                new_maps=all_new_maps)
            te_out, _ = func(Xy_te,
                             new_cat_feats=all_new_cat_feats,
                             new_maps=all_new_maps)
        else:
            tr_out, cats = func(Xy_tr, new_cat_feats=all_new_cat_feats)
            te_out, _ = func(Xy_te, new_cat_feats=all_new_cat_feats)

        X_tr = tr_out.drop(columns=["total_stats", "name"])
        y_tr = tr_out["total_stats"]
        X_te = te_out.drop(columns=["total_stats", "name"])
        y_te = te_out["total_stats"]

        data[name] = (X_tr, X_te, y_tr, y_te, cats)

    return data, fe_state


def prepare_data_predict(predict_df,
                         fe_state=None,
                         feat_eng_steps=None):
    """
    Prediction-data pipeline — mirrors prepare_data_train but:
      - No split (all rows are to be predicted).
      - Uses pre-computed fe_state from training (no refit).
      - Preserves 'name' column for output labelling.

    Args:
        predict_df:      Path string or DataFrame.
        fe_state:        Dict produced by prepare_data_train.
        feat_eng_steps:  Same list passed at train time.

    Returns:
        data (dict):  Keys = ["cat_ordinal", "cat_native", "light_gbm"].
                      Values = (X_pred_with_name, cats).
    """
    raw_df = load_data(predict_df)
    poke_names = load_data(predict_df)["name"]

    feat_eng_steps = feat_eng_steps or []
    fe_state = fe_state or {}

    all_new_cat_feats = []
    all_new_maps = {}

    for step_name, _, apply_func in feat_eng_steps:
        step_state = fe_state[step_name]
        raw_df = apply_func(raw_df, step_state)

        step_cat_feats = step_state.get("new_cat_feats", [])
        if step_cat_feats:
            all_new_cat_feats.extend(step_cat_feats)

        step_maps = step_state.get("new_maps", {})
        if step_maps:
            all_new_maps.update(step_maps)

    df_pred = cat_fill(raw_df)

    prep = {
        "cat_ordinal": prep_catboost_ordinal,
        "cat_native":  prep_catboost_native,
        "light_gbm":   prep_lightgbm,
    }

    data = {}
    for name, func in prep.items():
        if name == "cat_ordinal":
            out, cats = func(df_pred,
                             new_cat_feats=all_new_cat_feats,
                             new_maps=all_new_maps)
        else:
            out, cats = func(df_pred, new_cat_feats=all_new_cat_feats)

        out["name"] = poke_names.values
        data[name] = (out, cats)

    return data
