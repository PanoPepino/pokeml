# pokeml/classifier/trainer.py

import joblib
import pandas as pd

from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from pokeml.classifier.bst_bands import make_bst_band, band_prob_table


class BandClassifier:
    """
    Stage 1 classifier:
    Predicts a BST band before the regression model runs.

    Design choice:
    - Uses CatBoostClassifier because your project already supports
      categorical features well through CatBoost.
    - Expects a preprocessed X and the original numeric y (total_stats).
    - Uses the cat_native representation as the cleanest classifier input.
    """

    def __init__(self, **params):
        self.params = {
            "iterations": 5000,
            "learning_rate": 0.03,
            "depth": 6,
            "loss_function": "MultiClass",
            "eval_metric": "Accuracy",
            "auto_class_weights": "Balanced",
            "od_type": "Iter",
            "od_wait": 150,
            "verbose": 0,
            **params,
        }
        self.model = None
        self.classes_ = None

    def fit(self,
            X: pd.DataFrame,
            y_num: pd.Series,
            cat_features: list,
            full_df_for_labels: pd.DataFrame = None):
        """
        Fit the band classifier.

        Args:
            X:
                Feature dataframe already prepared for CatBoost native mode.
            y_num:
                Numeric total_stats target.
            cat_features:
                List of categorical feature names for CatBoost.
            full_df_for_labels:
                Optional DataFrame containing rarity/other metadata needed
                for the legendary_like override. If omitted, only numeric bins are used.
        """

        if full_df_for_labels is not None:
            y_band = make_bst_band(full_df_for_labels)
        else:
            y_band = make_bst_band(y_num)

        X_tr, X_va, y_tr, y_va = train_test_split(
            X,
            y_band,
            test_size=0.3,
            random_state=1,
            stratify=y_band,
        )

        train_pool = Pool(
            data=X_tr,
            label=y_tr.astype(str).tolist(),
            cat_features=cat_features,
        )
        valid_pool = Pool(
            data=X_va,
            label=y_va.astype(str).tolist(),
            cat_features=cat_features,
        )

        self.model = CatBoostClassifier(**self.params)
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        self.classes_ = list(self.model.classes_)
        return self

    def predict_band(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("BandClassifier model is not fitted yet")

        X_checked = self.validate_features(X)
        X_checked = self._coerce_cat_columns(X_checked)
        preds = self.model.predict(X_checked).ravel()
        return pd.Series(preds, index=X_checked.index, name="pred_band")

    def predict_proba_table(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("BandClassifier model is not fitted yet")

        X_checked = self.validate_features(X)
        X_checked = self._coerce_cat_columns(X_checked)
        return band_prob_table(self.model, X_checked)

    def enrich(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("BandClassifier model is not fitted yet")

        X_out = self.validate_features(X)
        X_out = self._coerce_cat_columns(X_out)

        pred_band = self.predict_band(X_out)
        proba_df = self.predict_proba_table(X_out)

        X_out = X_out.copy()
        X_out["pred_band"] = pred_band.values
        for col in proba_df.columns:
            X_out[f"proba_{col}"] = proba_df[col].values

        return X_out

    def save(self, path: str):
        """
        Save the full wrapper, not only the CatBoost model,
        so classes_ and params are preserved.
        """
        joblib.dump(self, Path(path), compress=3)

    @classmethod
    def load(cls, path: str):
        """
        Load a previously saved BandClassifier wrapper.
        """
        return joblib.load(Path(path))

    def validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Minimal validation hook.
        For now, just return a safe copy of X.
        """
        return X.copy()

    def _coerce_cat_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert known categorical columns to string so CatBoost accepts them.
        """
        Xc = X.copy()
        cat_cols = ["type_1", "type_2", "rarity", "stage", "shape", "color", "pred_band"]

        for col in cat_cols:
            if col in Xc.columns:
                Xc[col] = Xc[col].astype(str)

        return Xc
