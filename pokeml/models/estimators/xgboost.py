"""XGBoost regression with training-only one-hot encoding and quantile bounds."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor

from .common import _normalize_eval_set, validate_iterations


class XGB_Trainer(RegressorMixin, BaseEstimator):
    """Own fitted encoding so prediction and evaluation reuse training categories.

    Absolute-error point estimates align with pooled MAE. Quantile bounds express
    conditional uncertainty and remain uncalibrated; they exclude stage ambiguity.
    """

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, eval_set=None, **kwargs):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.cat_cols_ = list(X.select_dtypes(include=['object', 'string', 'category']).columns)
        numeric = [col for col in X if col not in self.cat_cols_]
        self.encoder_ = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_cols_),
            ('numeric', 'passthrough', numeric),
        ], sparse_threshold=0, verbose_feature_names_out=False)
        encoded = self.encoder_.fit_transform(X)
        self.encoded_feature_names_ = self.encoder_.get_feature_names_out()
        options = dict(n_estimators=400, max_depth=4, learning_rate=0.03,
                       tree_method='hist', n_jobs=1, random_state=1)
        options.update(self.params)
        fit_uncertainty = options.pop("fit_uncertainty", True)
        rounds = options.pop('early_stopping_rounds', None)
        eval_set = _normalize_eval_set(eval_set)
        evaluations = [(self.encoder_.transform(frame.loc[:, self.feature_names_in_]), target)
                       for frame, target in (eval_set or [])]
        if rounds and evaluations:
            options['early_stopping_rounds'] = rounds
        options.pop('objective', None)
        options.pop('eval_metric', None)
        options.pop('quantile_alpha', None)
        self.point_model_ = XGBRegressor(objective='reg:absoluteerror', eval_metric='mae', **options)
        fit_options = dict(kwargs)
        if evaluations:
            fit_options['eval_set'] = evaluations
        fit_options.setdefault('verbose', False)
        self.point_model_.fit(encoded, y, **fit_options)
        self.quantile_models_ = {}
        for quantile in ((0.1, 0.9) if fit_uncertainty else ()):
            model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile,
                                 eval_metric='quantile', **options)
            model.fit(encoded, y, **fit_options)
            self.quantile_models_[quantile] = model
        # Aggregate encoded columns back to original predictors for reporting.
        origins = []
        if self.cat_cols_:
            for column, categories in zip(self.cat_cols_, self.encoder_.named_transformers_['categorical'].categories_):
                origins.extend([column] * len(categories))
        origins.extend(numeric)
        self.importance_origins_ = np.asarray(origins, dtype=object)
        return self

    def _encoded(self, X):
        check_is_fitted(self, 'point_model_')
        return self.encoder_.transform(X.loc[:, self.feature_names_in_])

    def predict(self, X):
        encoded = self._encoded(X)
        return self.point_model_.predict(encoded)

    def predict_distribution(self, X):
        encoded = self._encoded(X)
        if not self.quantile_models_:
            raise ValueError("Quantile bounds unavailable: fit with fit_uncertainty=True")
        low = self.quantile_models_[0.1].predict(encoded)
        high = self.quantile_models_[0.9].predict(encoded)
        return {'pred_bst': self.point_model_.predict(encoded), 'lower_80': low,
                'upper_80': high, 'quantile_crossed': low > high}

    def predict_unc(self, X):
        result = self.predict_distribution(X)
        return result['pred_bst'], np.maximum(result['upper_80'] - result['lower_80'], 0) / 2

    def get_evals(self):
        check_is_fitted(self, 'point_model_')
        return self.point_model_.evals_result() if hasattr(self.point_model_, 'evals_result_') else {}

    def get_top_features(self, feature_names=None, k=5):
        check_is_fitted(self, 'point_model_')
        scores = np.asarray([self.point_model_.feature_importances_[self.importance_origins_ == col].sum()
                             for col in self.feature_names_in_], dtype=float)
        total = scores.sum()
        if total == 0:
            raise ValueError('XGBoost returned zero total feature importance.')
        scores *= 100 / total
        order = np.argsort(scores)[-k:][::-1]
        return self.feature_names_in_[order].tolist(), scores[order].tolist()

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        for key in list(vars(self)):
            if key.endswith('_'):
                delattr(self, key)
        return self

    def get_iteration_count(self):
        check_is_fitted(self, 'point_model_')
        return self.point_model_.get_booster().num_boosted_rounds()

    def predict_at_iterations(self, X, iterations):
        """Reuse training encoding while evaluating fitted boosting-round prefixes."""
        values = validate_iterations(iterations, self.get_iteration_count())
        encoded = self._encoded(X)
        for iteration in values:
            yield iteration, self.point_model_.predict(encoded, iteration_range=(0, int(iteration)))
