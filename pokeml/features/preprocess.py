# Here, I will define 3 different preprocessors to adapt data before model training.
# For convenience, I have also defined a function to check shapes match after preprocessing.

import pandas as pd
from sklearn.model_selection import train_test_split
from typer.cli import app

from pokeml.data.load import load_data


# First, we define a function to accommodate all data (to provide category flag + fill-in missing entries) to objects and also split X, y into train and test


def cat_fill(my_df):
    if isinstance(my_df, pd.DataFrame):
        new_df = my_df.copy()
    else:
        new_df = load_data(my_df).copy()

    new_df["type_2"] = new_df["type_2"].fillna("None")
    new_df[new_df.select_dtypes("object").columns] = (
        new_df.select_dtypes("object").astype("category")
    )

    return new_df

# We now define a preprocessor for catboost with ordinality transformation


def prep_catboost_ordinal(dataframe,
                          new_cat_feats=None,  # In case new categorical features appear
                          new_maps=None):  # To enhance if future feat. eng.
    """
    Preproccessor for catboostregressor. In this case, we create a map for two categories, to transform into simple numbers. 


    Args:
        dataframe (DataFrame): The Dataframe to be massaged. 
        new_cat_feats (list, optional): In case new categories appear in feat. eng.
        new_maps (dict, optional): In case previous new cats require a mapping, this is the place


    Returns:
        new_dataframe, cats. All preprocessed.
    """

    # Original maps
    maps = {'rarity': {'regular': 0,
                       'legendary': 1},
            'stage': {'s1c3': 1,
                      's1c2': 1.05,
                      's2c3': 1.47,
                      'single': 1.8,
                      's2c2': 1.85,
                      's3c3': 2}}

    # Regular cat feats
    cat_feats = ['type_1', 'type_2', 'shape', 'color']

    # In case of new maps
    if new_maps is not None:
        maps.update(new_maps)

    if new_cat_feats is not None:
        if isinstance(new_cat_feats, list):
            cat_feats.extend(new_cat_feats)
        else:
            cat_feats.append(new_cat_feats)

    new_dataframe = dataframe.copy()

    for col, mapping in maps.items():
        if col in new_dataframe.columns:
            new_dataframe[col] = new_dataframe[col].map(mapping).astype('float')

    return new_dataframe, cat_feats


# I will define now with catboost with natural, without any strange ordinality

def prep_catboost_native(dataframe,
                         new_cat_feats=None):  # To enhance if future feat. eng.
    """
    Another preprocessor function for catboost, but in this case native.


    Args:
        dataframe (DataFrame): The Dataframe to be massaged.
        new_cat_feats (list, optional): If new categories appear after feat. eng. Defaults to None.


    Returns:
        new_dataframe, cats. All preprocessed.
    """

    cat_feats = ['type_1', 'type_2', 'rarity', 'stage', 'shape', 'color']

    if new_cat_feats is not None:
        if isinstance(new_cat_feats, list):
            cat_feats.extend(new_cat_feats)
        else:
            cat_feats.append(new_cat_feats)

    # Transforming those columns to category
    new_dataframe = dataframe.copy()

    for col in cat_feats:
        if col in new_dataframe.columns:
            new_dataframe[col] = new_dataframe[col].astype('category')

    return new_dataframe, cat_feats


# Finally, time to define same preprocessor but for lightgbm

def prep_lightgbm(dataframe,
                  new_cat_feats=None):  # To enhance if future feat. eng.
    """
    Another preprocessor function for LGBM.


    Args:
        dataframe (DataFrame): The Dataframe to be massaged.
        new_cat_feats (list, optional): If new categories appear after feat. eng. Defaults to None.


    Returns:
        new_dataframe, cats. All preprocessed.
    """

    cat_feats = ['type_1', 'type_2', 'rarity', 'stage', 'shape', 'color']

    if new_cat_feats is not None:
        if isinstance(new_cat_feats, list):
            cat_feats.extend(new_cat_feats)
        else:
            cat_feats.append(new_cat_feats)

    # Transforming those columns to category
    new_dataframe = dataframe.copy()
    for col in cat_feats:
        if col in new_dataframe.columns:
            new_dataframe[col] = new_dataframe[col].astype('category')

    return new_dataframe, cat_feats


def prepare_data_train(df,
                       tsize=0.3,
                       feat_eng_steps=None):
    """
    This function will do 4 main things:


        - 1) Encode and impute with :func:`cat_fill_split`
        - 2) Preprocess all your data in 3 different ways with :func:`prep_name_model_mode`
        - 3) Split the train and test values with :func:`train_test_split`
        - 4) Spit out all processed data as dict.


    Args:
        df (my_df): the pokedex to use
        tsize (float): In case you want another ratio for the training set.


    Returns:
        data (Dict): All data processed according to preprocessors.
    """

    raw_df = load_data(df)

    fe_state = {}
    feat_eng_steps = feat_eng_steps or []

    all_new_cat_feats = []
    all_new_maps = {}

    for step_name, compute_func, apply_func in feat_eng_steps:
        step_state = compute_func(raw_df)
        fe_state[step_name] = step_state
        raw_df = apply_func(raw_df, step_state)

        step_cat_feats = step_state.get("new_cat_feats", [])
        if step_cat_feats:
            all_new_cat_feats.extend(step_cat_feats)

        step_maps = step_state.get("new_maps", {})
        print(step_maps)
        if step_maps:
            all_new_maps.update(step_maps)

    df_to_split = cat_fill(raw_df)
    Xy_tr, Xy_te = train_test_split(df_to_split, test_size=tsize, random_state=1)

    prep = {
        "cat_ordinal": prep_catboost_ordinal,
        "cat_native":  prep_catboost_native,
        "light_gbm": prep_lightgbm,
    }

    data = {}
    for name, func in prep.items():
        if name == "cat_ordinal":
            tr_out, cats = func(Xy_tr,
                                new_cat_feats=all_new_cat_feats,
                                new_maps=all_new_maps)
            te_out, _ = func(Xy_te,
                             new_cat_feats=all_new_cat_feats,
                             new_maps=all_new_maps)
        else:
            tr_out, cats = func(Xy_tr, new_cat_feats=all_new_cat_feats)
            te_out, _ = func(Xy_te, new_cat_feats=all_new_cat_feats)

        X_tr = tr_out.drop(columns=["total_stats", "name"])
        y_tr = tr_out["total_stats"]
        X_te = te_out.drop(columns=["total_stats", "name"])
        y_te = te_out["total_stats"]

        data[name] = (X_tr, X_te, y_tr, y_te, cats)

    return data, fe_state


def prepare_data_predict(predict_df,
                         fe_state=None,
                         feat_eng_steps=None):
    """
    This function will do 4 main things:


        - 1) Encode and impute with :func:`cat_fill_split`
        - 2) Preprocess all your data in 3 different ways with :func:`prep_name_model_mode`
        - 3) Split the train and test values with :func:`train_test_split`
        - 4) Spit out all processed data as dict.


    Args:
        df (my_df): the pokedex to use
        tsize (float): In case you want another ratio for the training set.


    Returns:
        data (Dict): All data processed according to preprocessors.
    """

    raw_df = load_data(predict_df)
    just_names = load_data(predict_df)
    poke_names = just_names['name']

    feat_eng_steps = feat_eng_steps or []
    fe_state = fe_state or {}

    all_new_cat_feats = []
    all_new_maps = {}

    for step_name, _, apply_func in feat_eng_steps:
        step_state = fe_state[step_name]
        raw_df = apply_func(raw_df, step_state)

        step_cat_feats = step_state.get("new_cat_feats", [])
        if step_cat_feats:
            all_new_cat_feats.extend(step_cat_feats)

        step_maps = step_state.get("new_maps", {})
        if step_maps:
            all_new_maps.update(step_maps)

    df_pred = cat_fill(raw_df)

    prep = {
        "cat_ordinal": prep_catboost_ordinal,
        "cat_native":  prep_catboost_native,
        "light_gbm": prep_lightgbm,
    }

    data = {}
    for name, func in prep.items():
        if name == "cat_ordinal":
            out, cats = func(df_pred,
                             new_cat_feats=all_new_cat_feats,
                             new_maps=all_new_maps)
        else:
            out, cats = func(df_pred, new_cat_feats=all_new_cat_feats)

        out['name'] = poke_names.values
        data[name] = (out, cats)

    return data
