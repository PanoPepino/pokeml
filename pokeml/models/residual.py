"""One sklearn estimator owns baseline fitting, feature state and BST reconstruction."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from pokeml.models.baseline import StageMedianBaseline, validate_rows
from pokeml.models.factory import make_estimator
from pokeml.pipeline.transformer import ModelPreprocessor
from pokeml.pipeline.schema import BASE_FEATURES


class BaselineCorrectionRegressor(RegressorMixin, BaseEstimator):
    """Fit raw rows, subtract training baseline, and predict BST rather than residual.

    The direct mode supplies an identical-feature comparison. Feature transforms
    receive original BST targets, never residual targets. Supervised type maps
    remain excluded from this first experiment. No headline holdout is passed
    to the estimator for early stopping or iteration selection.
    """

    def __init__(self, variant="cat_native", estimator=None, steps=None,
                 min_count=5, random_state=1, mode="correction"):
        self.variant = variant
        self.estimator = estimator
        self.steps = steps
        self.min_count = min_count
        self.random_state = random_state
        self.mode = mode

    def _validate(self, X):
        validate_rows(X)
        missing = set(BASE_FEATURES) - set(X)
        if missing:
            raise ValueError(f"Missing announcement attributes: {sorted(missing)}")
        for column in ("height", "weight"):
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{column} must be finite and nonnegative (metres / kilograms)")
        return X

    def fit(self, X, y):
        self._validate(X)
        if self.mode not in ("direct", "correction"):
            raise ValueError("mode must be direct or correction")
        if any(step[0] == "type_deviation" for step in (self.steps or [])):
            raise ValueError("Target-derived type maps are disabled for baseline + correction v1")
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(target) != len(X) or not np.isfinite(target).all():
            raise ValueError("Targets must be finite and match input row count")
        self.baseline_ = StageMedianBaseline(self.min_count).fit(X, target)
        self.preprocessor_ = ModelPreprocessor(self.variant, self.steps, self.random_state)
        encoded = self.preprocessor_.fit_transform(X, target)
        self.feature_names_in_ = np.asarray(encoded.columns, dtype=object)
        self.model_ = clone(self.estimator) if self.estimator is not None else make_estimator(self.variant)
        if self.variant.startswith("cat_"):
            self.model_.set_params(cat_features=self.preprocessor_.cat_features_)
        options = self.model_.get_params()
        if options.get("early_stopping_rounds"):
            raise ValueError("Use fixed iteration counts; headline holdout cannot select iterations")
        if self.variant.startswith("cat_"):
            self.model_.set_params(use_best_model=False)
        baseline = self.baseline_.predict(X) if self.mode == "correction" else np.zeros(len(X))
        self.training_residuals_ = target - baseline
        self.model_.fit(encoded, self.training_residuals_)
        return self

    def _inputs(self, X):
        check_is_fitted(self, "model_")
        self._validate(X)
        return self.preprocessor_.transform(X).loc[:, self.feature_names_in_]

    def predict_components(self, X):
        encoded = self._inputs(X)
        result = self.baseline_.predict_components(X)
        output = self.model_.predict(encoded)
        # Direct mode reports a post-hoc difference from the reference; its model
        # was trained on BST, not residuals. Expose mode rather than hide that fact.
        result["correction_bst"] = output if self.mode == "correction" else output - result.baseline_bst
        result["pred_bst"] = result.baseline_bst + result.correction_bst
        result["prediction_mode"] = self.mode
        return result

    def predict(self, X):
        return self.predict_components(X).pred_bst.to_numpy()

    def predict_distribution(self, X):
        encoded = self._inputs(X)
        distribution = self.model_.predict_distribution(encoded)
        baseline = self.baseline_.predict(X) if self.mode == "correction" else np.zeros(len(X))
        # A deterministic location shift moves point/quantile values, not variances.
        return {key: np.asarray(value) + baseline if key in ("pred_bst", "lower_80", "upper_80")
                else value for key, value in distribution.items()}

    def get_top_features(self, k=5):
        check_is_fitted(self, "model_")
        return self.model_.get_top_features(self.feature_names_in_, k=k)

    def get_iteration_count(self):
        check_is_fitted(self, "model_")
        return self.model_.get_iteration_count()

    def predict_at_iterations(self, X, iterations):
        """Replay fitted preprocessing once and reconstruct BST at each prefix."""
        encoded = self._inputs(X)
        baseline = self.baseline_.predict(X) if self.mode == "correction" else np.zeros(len(X))
        for iteration, prediction in self.model_.predict_at_iterations(encoded, iterations):
            yield iteration, np.asarray(prediction) + baseline
