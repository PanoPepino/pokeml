import pandas as pd
from pokeml.data.load import load_data


# First, we define a function to accommodate all data (to provide category flag + fill-in missing entries) to objects and also split X, y into train and test

def cat_fill(my_df) -> pd.DataFrame:
    """
    Accept either a raw DataFrame or a file path string.
    Does two things:
      1. Fills missing type_2 with the string "None" so CatBoost
         and LGBM never see NaN in a categorical column.
      2. Casts all remaining object columns to pandas Categorical,
         which is required before any of the prep_* functions run.

    Args:
        my_df: pd.DataFrame already in memory, or a str path to CSV.

    Returns:
        A cleaned copy — original is never mutated.
    """
    if isinstance(my_df, pd.DataFrame):
        new_df = my_df.copy()
    else:
        new_df = load_data(my_df).copy()

    new_df["type_2"] = new_df["type_2"].fillna("None")
    new_df[new_df.select_dtypes("object").columns] = (
        new_df.select_dtypes("object").astype("category")
    )
    return new_df
