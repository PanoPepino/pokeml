import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.base import BaseEstimator, RegressorMixin
from .common import _normalize_eval_set, validate_iterations

class LGBM_Trainer(RegressorMixin, BaseEstimator):
    def __init__(self, **params):
        self.params = params

        self.median_model = None
        self.quantile_models = None
        self.cat_cols_ = None

    def _prepare_lgbm_frames(self, x, eval_set=None):
        x = x.copy()

        cat_cols = [
            c for c in x.columns
            if pd.api.types.is_string_dtype(x[c]) or isinstance(x[c].dtype, pd.CategoricalDtype)
        ]

        for col in cat_cols:
            x[col] = x[col].astype("category")

        eval_set = _normalize_eval_set(eval_set)

        eval_set_lgbm = None
        if eval_set is not None:
            eval_set_lgbm = []
            for X_eval, y_eval in eval_set:
                X_eval = X_eval.copy()
                for col in cat_cols:
                    if col in X_eval.columns:
                        X_eval[col] = pd.Categorical(
                            X_eval[col],
                            categories=x[col].cat.categories
                        )
                eval_set_lgbm.append((X_eval, y_eval))

        return x, eval_set_lgbm, cat_cols

    def fit(self, x, y, eval_set=None, **kwargs):
        x, eval_set_lgbm, cat_cols = self._prepare_lgbm_frames(x, eval_set=eval_set)
        self.cat_cols_ = cat_cols
        self.categories_ = {col: x[col].cat.categories for col in cat_cols}
        options = {"n_estimators": 2500, "learning_rate": 0.01, "verbosity": -1,
                         "random_state": 1, "n_jobs": 1}
        options.update(self.params)
        fit_uncertainty = options.pop("fit_uncertainty", True)
        rounds = options.pop("early_stopping_rounds", None)
        def fit_options():
            result = dict(kwargs)
            callbacks = list(result.pop("callbacks", []))
            if rounds and eval_set_lgbm:
                callbacks.append(early_stopping(rounds, verbose=False))
            if callbacks:
                result["callbacks"] = callbacks
            return result

        self.median_model = LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            **options
        )
        self.median_model.fit(
            x,
            y,
            eval_set=eval_set_lgbm,
            eval_metric="quantile",
            categorical_feature=cat_cols,
            **fit_options()
        )

        quants = [0.1, 0.9]
        self.quantile_evals_ = {}
        self.quantile_models = {}

        for quant in (quants if fit_uncertainty else []):
            model_q = LGBMRegressor(
                objective="quantile",
                alpha=quant,
                **options
            )
            model_q.fit(
                x,
                y,
                eval_set=eval_set_lgbm,
                eval_metric="quantile",
                categorical_feature=cat_cols,
                **fit_options(),
            )
            self.quantile_models[quant] = model_q
            self.quantile_evals_[quant] = model_q.evals_result_

        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        return self

    def __sklearn_is_fitted__(self):
        return self.median_model is not None

    def get_evals(self):
        if self.median_model is None:
            raise ValueError("LGBM_Trainer is not fitted yet.")
        return self.median_model.evals_result_

    def _prepare_predict_frame(self, X):
        if self.cat_cols_ is None:
            return X.copy()

        X = X.copy()
        for col in self.cat_cols_:
            if col in X.columns:
                if hasattr(self, "categories_"):
                    X[col] = pd.Categorical(X[col].astype(object).where(X[col].isin(self.categories_[col])),
                                            categories=self.categories_[col])
                else:
                    # Legacy joblib models let LightGBM replay stored pandas categories.
                    X[col] = X[col].astype("category")
        return X

    def predict(self, X):
        if self.median_model is None:
            raise ValueError("LGBM_Trainer is not fitted yet.")
        X = self._prepare_predict_frame(X)
        return self.median_model.predict(X)

    def predict_unc(self, X):
        if not self.quantile_models:
            raise ValueError("LGBM_Trainer is not fitted yet.")
        X = self._prepare_predict_frame(X)
        low = self.quantile_models[0.1].predict(X)
        high = self.quantile_models[0.9].predict(X)
        return self.predict(X), np.maximum(high - low, 0) / 2

    def predict_distribution(self, X):
        """Median point estimate and uncalibrated 10th/90th quantile bounds."""
        if not self.quantile_models:
            raise ValueError("LGBM_Trainer is not fitted yet.")
        X = self._prepare_predict_frame(X)
        low = self.quantile_models[0.1].predict(X)
        high = self.quantile_models[0.9].predict(X)
        return {"pred_bst": self.predict(X), "lower_80": low, "upper_80": high,
                "quantile_crossed": low > high}

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        self.median_model = None
        self.quantile_models = None
        self.cat_cols_ = None
        return self

    def get_top_features(self, feature_names, k=5):
        if self.median_model is None:
            raise ValueError("LGBM_Trainer is not fitted yet.")

        raw_importances = np.asarray(self.median_model.feature_importances_, dtype=float)
        total = raw_importances.sum()

        if total == 0:
            raise ValueError("LightGBM returned zero total feature importance.")

        norm_importances = 100.0 * raw_importances / total
        top_idx = np.argsort(norm_importances)[-k:][::-1]

        top_features = [feature_names[i] for i in top_idx]
        top_importances = [float(norm_importances[i]) for i in top_idx]
        return top_features, top_importances

    def get_iteration_count(self):
        if self.median_model is None:
            raise ValueError("LGBM_Trainer is not fitted yet.")
        return self.median_model.booster_.current_iteration()

    def predict_at_iterations(self, X, iterations):
        """Yield median predictions using actual fitted boosting-round prefixes."""
        values = validate_iterations(iterations, self.get_iteration_count())
        frame = self._prepare_predict_frame(X)
        for iteration in values:
            yield iteration, self.median_model.predict(frame, num_iteration=int(iteration))
