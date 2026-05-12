import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin


def _normalize_eval_set(eval_set):
    if eval_set is None:
        return None

    if isinstance(eval_set, tuple) and len(eval_set) == 2:
        return [eval_set]

    if isinstance(eval_set, list):
        return eval_set

    raise ValueError(
        "eval_set must be None, a tuple (X, y), or a list of tuples [(X1, y1), ...]."
    )


class Cat_Trainer(BaseEstimator, RegressorMixin):
    def __init__(self, cat_features=None, **params):
        self.cat_features = cat_features or []
        self.params = {
            "loss_function": "RMSEWithUncertainty",
            "eval_metric": "RMSEWithUncertainty",
            "posterior_sampling": True,
            "learning_rate": 0.01,
            "verbose": 0,
            **params
        }
        self.model = None

    def fit(self, x, y, eval_set=None, **kwargs):
        x = x.copy()

        cat_cols = [c for c in self.cat_features if c in x.columns]
        for col in cat_cols:
            x[col] = x[col].astype(str)

        eval_set = _normalize_eval_set(eval_set)

        eval_set_cb = None
        if eval_set is not None:
            eval_set_cb = []
            for X_eval, y_eval in eval_set:
                X_eval = X_eval.copy()
                for col in cat_cols:
                    if col in X_eval.columns:
                        X_eval[col] = X_eval[col].astype(str)
                eval_set_cb.append((X_eval, y_eval))

        self.model = CatBoostRegressor(
            cat_features=cat_cols,
            **self.params
        )
        self.model.fit(
            x,
            y,
            eval_set=eval_set_cb,
            **kwargs
        )
        return self

    def get_evals(self):
        if self.model is None:
            raise ValueError("Cat_Trainer is not fitted yet.")
        return self.model.get_evals_result()

    def predict(self, test_data):
        if self.model is None:
            raise ValueError("Cat_Trainer is not fitted yet.")

        X = test_data.copy()
        cat_cols = [c for c in self.cat_features if c in X.columns]
        for col in cat_cols:
            X[col] = X[col].astype(str)

        preds = self.model.predict(X)
        if np.ndim(preds) == 2:
            return preds[:, 0]
        return preds

    def predict_unc(self, test_data):
        if self.model is None:
            raise ValueError("Cat_Trainer is not fitted yet.")

        X = test_data.copy()
        cat_cols = [c for c in self.cat_features if c in X.columns]
        for col in cat_cols:
            X[col] = X[col].astype(str)

        val_unc = self.model.virtual_ensembles_predict(
            X,
            prediction_type="TotalUncertainty"
        )
        vals = val_unc[:, 0]
        uncs = val_unc[:, 1]
        return vals, uncs

    def get_params(self, deep=True):
        tunable_params = {
            "learning_rate": self.params.get("learning_rate"),
            "max_depth": self.params.get("max_depth"),
            "iterations": self.params.get("iterations"),
            "l2_leaf_reg": self.params.get("l2_leaf_reg"),
            "random_strength": self.params.get("random_strength"),
            "bagging_temperature": self.params.get("bagging_temperature"),
            "n_estimators": self.params.get("n_estimators"),
            "early_stopping_rounds": self.params.get("early_stopping_rounds"),
        }
        tunable_params = {k: v for k, v in tunable_params.items() if v is not None}
        return {"cat_features": self.cat_features, **tunable_params}

    def set_params(self, **params):
        if "cat_features" in params:
            self.cat_features = params.pop("cat_features")
        self.params.update(params)
        self.model = None
        return self

    def get_top_features(self, feature_names, k=4):
        if self.model is None:
            raise ValueError("Cat_Trainer is not fitted yet.")

        importances = self.model.get_feature_importance()
        top_idx = np.argsort(importances)[-k:][::-1]

        top_features = [feature_names[i] for i in top_idx]
        top_importances = [float(importances[i]) for i in top_idx]
        return top_features, top_importances


class LGBM_Trainer(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self.params = {
            "n_estimators": 2500,
            "learning_rate": 0.01,
            "verbosity": -1,
            **params
        }
        self.median_model = None
        self.quantile_models = None
        self.cat_cols_ = None

    def _prepare_lgbm_frames(self, x, eval_set=None):
        x = x.copy()

        cat_cols = [
            c for c in x.columns
            if pd.api.types.is_string_dtype(x[c]) or pd.api.types.is_categorical_dtype(x[c])
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

        self.median_model = LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            **self.params
        )
        self.median_model.fit(
            x,
            y,
            eval_set=eval_set_lgbm,
            eval_metric="quantile",
            categorical_feature=cat_cols,
            **kwargs
        )

        quants = [0.1, 0.9]
        self.quantile_evals_ = {}
        self.quantile_models = {}

        for quant in quants:
            model_q = LGBMRegressor(
                objective="quantile",
                alpha=quant,
                **self.params
            )
            model_q.fit(
                x,
                y,
                eval_set=eval_set_lgbm,
                eval_metric="quantile",
                categorical_feature=cat_cols,
            )
            self.quantile_models[quant] = model_q
            self.quantile_evals_[quant] = {}

        return self

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
                X[col] = X[col].astype("category")
        return X

    def predict(self, X):
        if self.median_model is None:
            raise ValueError("LGBM_Trainer is not fitted yet.")
        X = self._prepare_predict_frame(X)
        return self.median_model.predict(X)

    def predict_unc(self, X):
        if self.quantile_models is None:
            raise ValueError("LGBM_Trainer is not fitted yet.")
        X = self._prepare_predict_frame(X)
        low = self.quantile_models[0.1].predict(X)
        high = self.quantile_models[0.9].predict(X)
        return (low + high) / 2, (high - low) / 2

    def get_params(self, deep=True):
        tunable = {
            k: v for k, v in self.params.items()
            if k in ["learning_rate", "n_estimators", "num_leaves", "reg_lambda"]
        }
        return tunable

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
