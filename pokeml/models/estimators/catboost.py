import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from .common import _normalize_eval_set, validate_iterations

class Cat_Trainer(RegressorMixin, BaseEstimator):
    def __init__(self, cat_features=None, **params):
        self.cat_features = cat_features
        self.params = params

        self.model = None

    def fit(self, x, y, eval_set=None, **kwargs):
        x = x.copy()

        cat_cols = [c for c in (self.cat_features or []) if c in x.columns]
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

        options = {"loss_function": "RMSEWithUncertainty", "eval_metric": "RMSEWithUncertainty",
                         "posterior_sampling": True, "learning_rate": 0.01, "verbose": 0,
                         "random_seed": 1, "thread_count": 1, "allow_writing_files": False}
        options.update(self.params)
        self.model = CatBoostRegressor(
            cat_features=cat_cols,
            **options
        )
        self.model.fit(
            x,
            y,
            eval_set=eval_set_cb,
            **kwargs
        )
        self.n_features_in_ = x.shape[1]
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        return self

    def __sklearn_is_fitted__(self):
        return self.model is not None

    def get_evals(self):
        if self.model is None:
            raise ValueError("Cat_Trainer is not fitted yet.")
        return self.model.get_evals_result()

    def predict(self, test_data):
        if self.model is None:
            raise ValueError("Cat_Trainer is not fitted yet.")

        X = test_data.copy()
        cat_cols = [c for c in (self.cat_features or []) if c in X.columns]
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
        cat_cols = [c for c in (self.cat_features or []) if c in X.columns]
        for col in cat_cols:
            X[col] = X[col].astype(str)

        val_unc = self.model.virtual_ensembles_predict(
            X,
            prediction_type="TotalUncertainty"
        )
        vals = val_unc[:, 0]
        # Total variance = knowledge variance + data variance. Return BST-unit sigma.
        uncs = np.sqrt(np.maximum(val_unc[:, 1:].sum(axis=1), 0))
        return vals, uncs

    def predict_distribution(self, X):
        """Expose variance components; sigma is not a calibrated prediction interval."""
        if self.model is None:
            raise ValueError("Cat_Trainer is not fitted yet.")
        X = X.copy()
        for col in self.cat_features or []:
            if col in X:
                X[col] = X[col].astype(str)
        values = self.model.virtual_ensembles_predict(X, prediction_type="TotalUncertainty")
        if values.shape[1] != 3:
            raise ValueError("Distribution output requires RMSEWithUncertainty loss")
        return {"pred_bst": self.predict(X), "knowledge_variance": values[:, 1],
                "data_variance": values[:, 2],
                "predictive_std": np.sqrt(np.maximum(values[:, 1:].sum(axis=1), 0))}

    def get_params(self, deep=True):
        return {"cat_features": self.cat_features, **self.params}

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



    def get_iteration_count(self):
        if self.model is None:
            raise ValueError("Cat_Trainer is not fitted yet.")
        return self.model.tree_count_

    def predict_at_iterations(self, X, iterations):
        """Yield point predictions from fixed fitted tree prefixes."""
        values = validate_iterations(iterations, self.get_iteration_count())
        X = X.copy()
        for col in self.cat_features or []:
            if col in X:
                X[col] = X[col].astype(str)
        wanted = set(values)
        stages = self.model.staged_predict(X, ntree_end=values[-1], eval_period=1,
                                           thread_count=1)
        for iteration, prediction in enumerate(stages, start=1):
            if iteration in wanted:
                prediction = np.asarray(prediction)
                yield iteration, prediction[:, 0] if prediction.ndim == 2 else prediction
