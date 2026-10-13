# pokeml/pipeline/prepare.py
# Responsible for: orchestrating feature engineering, cleaning,
# splitting, and packaging data for all four model variants.
# The four prep_* encoding functions live in features/preprocess.py.
# This layer calls them — it does not define them.

from sklearn.model_selection import train_test_split

from pokeml.data.load import load_data
from pokeml.data.catfill import cat_fill
from pokeml.features.engine import FeatureEngine
from pokeml.features.preprocess import prep_catboost_ordinal, prep_catboost_native, prep_lightgbm
from pokeml.pipeline.schema import BASE_FEATURES, TARGET, model_frame

ENCODERS = {
    "cat_ordinal": prep_catboost_ordinal,
    "cat_native": prep_catboost_native,
    "light_gbm": prep_lightgbm,
    # XGBoost wrapper learns one-hot categories on training rows only.
    "xgboost": prep_catboost_native,
}


class PreparedData(dict):
    """Original mapping contract plus raw training rows for fold-local CV.

    Existing indexing/unpacking remains valid. Tuning can reconstruct preprocessing
    inside CV rather than reuse features computed outside its folds.
    """


def encode_frame(frame, variant, engine):
    """Encode one frame using shared feature metadata and an explicit input schema."""
    engineered = [c for c in frame if c not in BASE_FEATURES and c not in
                  ("name", TARGET, "id", "generation", "species_id", "evolution_chain_id")]
    # Only columns actually introduced by the engine may extend the base schema.
    original_columns = getattr(engine, "input_columns", set())
    engineered = [c for c in engineered if c not in original_columns]
    selected = model_frame(frame, engineered)
    kwargs = {"new_cat_feats": engine.categorical_features}
    if variant == "cat_ordinal":
        kwargs["new_maps"] = engine.maps
    return ENCODERS[variant](cat_fill(selected), **kwargs)


def prepare_data_split(train_df, validation_df, feat_eng_steps=None, *,
                       random_state=1, groups=None):
    """Prepare explicitly chosen cohorts without fitting on validation targets."""
    train_df, validation_df = load_data(train_df), load_data(validation_df)
    for frame in (train_df, validation_df):
        if frame[TARGET].isna().any():
            raise ValueError("Training and validation targets must be present")
    engine = FeatureEngine(feat_eng_steps)
    engine.input_columns = set(train_df.columns)
    transformed_train = engine.fit_transform(train_df, groups=groups, random_state=random_state)
    transformed_validation = engine.transform(validation_df)
    data = PreparedData()
    for variant in ENCODERS:
        X_train, cats = encode_frame(transformed_train, variant, engine)
        X_valid, _ = encode_frame(transformed_validation, variant, engine)
        data[variant] = (X_train, X_valid, train_df[TARGET].copy(),
                         validation_df[TARGET].copy(), cats)
    data.raw_train = train_df.copy()
    data.raw_validation = validation_df.copy()
    data.feature_steps = list(feat_eng_steps or [])
    data.random_state = random_state
    data.groups = groups
    return data, engine.state


def prepare_data_train(df, tsize=0.3, feat_eng_steps=None, *, random_state=1,
                       validation_df=None, validation_generation=None):
    """
    Full training-data pipeline:
      1. Load raw data.
      2. Split before learning any feature state.
      3. Fit feature state on training only; cross-fit target-derived features.
      4. Clean and cast each split with cat_fill.
      5. Run all four prep_* encoders.
      6. Return a dict keyed by model variant + the fe_state.

    Args:
        df:              Path string or DataFrame.
        tsize:           Test fraction (default 0.3).
        feat_eng_steps:  List of (step_name, compute_fn, apply_fn) tuples.
        random_state:    Seed shared by splitting and supervised-feature cross-fitting.
        validation_df:   Optional explicit validation cohort (no automatic split).
        validation_generation: Optional generation holdout; train uses earlier generations.

    Returns:
        data (dict):  Keys = ["cat_ordinal", "cat_native", "light_gbm", "xgboost"].
                      Values = (X_tr, X_te, y_tr, y_te, cats).
        fe_state (dict): Computed state from each feature engineering step,
                         needed to replay the same transforms at predict time.
    """
    raw_df = load_data(df)
    if validation_df is not None and validation_generation is not None:
        raise ValueError("Choose validation_df or validation_generation, not both")
    if validation_df is not None:
        train_df, valid_df = raw_df, load_data(validation_df)
    elif validation_generation is not None:
        if "generation" not in raw_df:
            raise ValueError("Generation split requires a dataset with generation metadata")
        train_df = raw_df.loc[raw_df.generation < validation_generation]
        valid_df = raw_df.loc[raw_df.generation == validation_generation]
    else:
        train_df, valid_df = train_test_split(raw_df, test_size=tsize, random_state=random_state)
    if train_df.empty or valid_df.empty:
        raise ValueError("Training and validation cohorts must both contain rows")
    return prepare_data_split(train_df, valid_df, feat_eng_steps, random_state=random_state)
