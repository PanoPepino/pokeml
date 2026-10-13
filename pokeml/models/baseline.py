"""Fitted training-only stage references, shared by evaluation and serving."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

STAGES = ("single", "s1c2", "s2c2", "s1c3", "s2c3", "s3c3")
RARITIES = ("regular", "legendary")


def validate_rows(rows):
    """Require supplied stage/rarity; never silently infer announcement metadata."""
    if not isinstance(rows, pd.DataFrame) or rows.empty:
        raise ValueError("Provide a nonempty pandas DataFrame")
    for column, allowed in (("stage", STAGES), ("rarity", RARITIES)):
        if column not in rows:
            raise ValueError(f"Missing required input: {column}")
        if not rows[column].isin(allowed).all():
            raise ValueError(f"Invalid or missing {column}; allowed values: {allowed}")
    return rows


class StageMedianBaseline(RegressorMixin, BaseEstimator):
    """Bucket median → rarity median → global median, fitted on training rows only.

    A bucket needs min_count rows. Fallback estimates remain visible in outputs;
    outliers stay in training data. This is a conditional point reference, not
    an announcement-time evolution classifier or a calibrated interval.
    """

    def __init__(self, min_count=5):
        self.min_count = min_count

    def fit(self, X, y):
        validate_rows(X)
        if not isinstance(self.min_count, int) or self.min_count < 1:
            raise ValueError("min_count must be a positive integer")
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(target) != len(X) or not np.isfinite(target).all():
            raise ValueError("Targets must be finite and match input row count")
        frame = X[["stage", "rarity"]].copy()
        frame["total_stats"] = target
        self.buckets_ = frame.groupby(["rarity", "stage"], observed=True).total_stats.agg(
            median="median", count="size")
        self.rarity_medians_ = frame.groupby("rarity", observed=True).total_stats.median()
        self.global_median_ = float(np.median(target))
        return self

    def predict_components(self, X):
        check_is_fitted(self, "buckets_")
        validate_rows(X)
        values, sources, counts = [], [], []
        for rarity, stage in zip(X.rarity, X.stage):
            key = (rarity, stage)
            count = int(self.buckets_.loc[key, "count"]) if key in self.buckets_.index else 0
            if count >= self.min_count:
                value, source = float(self.buckets_.loc[key, "median"]), "stage_rarity"
            elif rarity in self.rarity_medians_.index:
                value, source = float(self.rarity_medians_.loc[rarity]), "rarity"
            else:
                value, source = self.global_median_, "global"
            values.append(value)
            sources.append(source)
            counts.append(count)
        return pd.DataFrame({"baseline_bst": values, "correction_bst": 0.0,
                             "pred_bst": values, "baseline_source": sources,
                             "baseline_count": counts, "prediction_mode": "baseline"}, index=X.index)

    def predict(self, X):
        return self.predict_components(X).pred_bst.to_numpy()

    def predict_distribution(self, X):
        return {"pred_bst": self.predict(X)}
