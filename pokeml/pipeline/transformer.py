"""Scikit-learn bridge: feature fitting stays inside each search fold."""

from sklearn.base import BaseEstimator, TransformerMixin

from pokeml.data.load import load_data
from pokeml.features.engine import FeatureEngine
from pokeml.pipeline.prepare import encode_frame


class ModelPreprocessor(TransformerMixin, BaseEstimator):
    """Transform raw rows for one estimator, cross-fitting supervised features."""

    def __init__(self, variant="cat_native", steps=None, random_state=1):
        self.variant = variant
        self.steps = steps
        self.random_state = random_state

    def _frame(self, X, y=None):
        frame = load_data(X)
        if y is not None:
            frame["total_stats"] = y.to_numpy() if hasattr(y, "to_numpy") else y
        return frame

    def fit(self, X, y=None):
        frame = self._frame(X, y)
        self.engine_ = FeatureEngine(self.steps)
        self.engine_.input_columns = set(frame.columns)
        transformed = self.engine_.fit(frame).transform(frame)
        encoded, self.cat_features_ = encode_frame(transformed, self.variant, self.engine_)
        self.feature_names_ = list(encoded.columns)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        frame = self._frame(X, y)
        self.engine_ = FeatureEngine(self.steps)
        self.engine_.input_columns = set(frame.columns)
        transformed = self.engine_.fit_transform(frame, random_state=self.random_state)
        encoded, self.cat_features_ = encode_frame(transformed, self.variant, self.engine_)
        self.feature_names_ = list(encoded.columns)
        return encoded

    def transform(self, X):
        encoded, _ = encode_frame(self.engine_.transform(self._frame(X)),
                                  self.variant, self.engine_)
        return encoded

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_
