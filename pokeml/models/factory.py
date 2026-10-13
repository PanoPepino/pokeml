"""Single place to resolve variant names into estimator wrappers."""

from pokeml.models.estimators import Cat_Trainer, LGBM_Trainer, XGB_Trainer


def make_estimator(variant, params=None, categorical_features=None):
    params = dict(params or {})
    if variant in ("cat_native", "cat_ordinal"):
        configured_cats = params.pop("cat_features", None)
        cats = categorical_features if categorical_features is not None else configured_cats
        return Cat_Trainer(cat_features=cats, **params)
    if variant == "light_gbm":
        return LGBM_Trainer(**params)
    if variant == "xgboost":
        return XGB_Trainer(**params)
    raise ValueError(f"Unknown model variant: {variant}")
