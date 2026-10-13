"""Local artifact paths and lightweight readers; no data acquisition."""

MODEL_NAMES = ("cat_native", "cat_ordinal", "light_gbm", "xgboost")





def tuning_section(variant):
    """Resolve each variant to its own tuning grid."""
    if variant not in MODEL_NAMES:
        raise ValueError(f"Unknown model variant: {variant}")
    return "catboost" if variant.startswith("cat_") else variant
