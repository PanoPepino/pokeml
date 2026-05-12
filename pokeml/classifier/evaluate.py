# pokeml/bandclassifier/bst_bands.py

from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

from pokeml.constants import BST_BINS, BST_LABELS, FINAL_LABELS


def make_bst_band(df_or_y, rarity_col: str = "rarity") -> pd.Series:
    """
    Convert total_stats into BST band labels.

    Supports:
    - a Series of numeric total_stats
    - a DataFrame with 'total_stats' and optionally 'rarity'

    Logic:
    1. First assign a pure numeric BST band via pd.cut.
    2. Then optionally override rows tagged as legendary into 'legendary_like'.

    Args:
        df_or_y:
            Either a numeric pandas Series, or a DataFrame containing total_stats.
        rarity_col:
            Name of the rarity column when a DataFrame is passed.

    Returns:
        Ordered categorical Series with FINAL_LABELS.
    """

    if isinstance(df_or_y, pd.Series):
        y = df_or_y.copy()
        band = pd.cut(y, bins=BST_BINS, labels=BST_LABELS).astype(str)
        return pd.Series(
            pd.Categorical(band, categories=FINAL_LABELS, ordered=True),
            index=y.index,
            name="bst_band",
        )

    if isinstance(df_or_y, pd.DataFrame):
        if "total_stats" not in df_or_y.columns:
            raise KeyError("'total_stats' column is required to build BST bands")

        band = pd.cut(
            df_or_y["total_stats"],
            bins=BST_BINS,
            labels=BST_LABELS,
        ).astype(str)

        if rarity_col in df_or_y.columns:
            legendary_mask = (
                df_or_y[rarity_col]
                .astype(str)
                .str.contains("legend", case=False, na=False)
            )
            band = np.where(legendary_mask, "legendary_like", band)

        return pd.Series(
            pd.Categorical(band, categories=FINAL_LABELS, ordered=True),
            index=df_or_y.index,
            name="bst_band",
        )

    raise TypeError("make_bst_band expects either a pandas Series or DataFrame")


def band_prob_table(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Turn classifier probabilities into a labeled DataFrame.
    """
    proba = model.predict_proba(X)
    return pd.DataFrame(proba, columns=model.classes_, index=X.index)


def band_report(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Print simple classification metrics for Stage 1.
    """
    print("Band accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(classification_report(y_true, y_pred, zero_division=0))
