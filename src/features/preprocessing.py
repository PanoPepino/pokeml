# Here, I will define 4 different preprocessors to adapt data before model training.
# For convenience, I have also defined a function to check shapes match after preprocessing.
# At the moment prep_xgboost_ohe does not work when running the model fit function.
# In any case, that does not seem like a big deal, as CatBoost seems to yield a better fitting.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(df, new_poke_info):
    """
    This function will do 4 main things:

        - 1) Drop values from X, y, encode and impute with :func:`cat_fill_split`
        - 2) Split the train and test values with :func:`train_test_split`
        - 3) Preprocess all your data in 3 different ways with :func:`prep_name_model_mode`
        - 4) Spit out all processed data as dict.

    Args:
        df (DataFrame): the pokedex to use
        new_poke_info (DataFrame): An extension of the new poke information.

    Returns:
        data (Dict): All data processed according to preprocessors.
    """

    # 1: Drop BST values, encode and impute and split df for training
    X, y = df.drop(columns="total_stats"), df["total_stats"]
    X_train, X_test, y_train, y_test, new_gen_values = cat_fill_split(df, new_poke_info)
    X_tr, X_te, y_tr, y_te = train_test_split(X_train.copy(), y_train.copy(), test_size=0.25, random_state=1)

    # 2: Drop the name of the new_pokemon_info. Store names in list for display use
    gen_names = new_poke_info["name"].tolist()
    gen_feats = new_poke_info.drop(columns="name")

    # 3: Here the dictionariony with all preprocessors. In case you add one more, do not forget to add here.
    prep = {
        "ordinal": prep_catboost_ordinal,
        "native":  prep_catboost_native,
        "light_gbm": prep_lightgbm,
    }

    # 4: Data will be output in a dic.
    data = {}
    for name, func in prep.items():  # dict.items will open and unzip keys and values!
        # func act as the value of the dict and map it.
        X_tr_p, X_te_p, gen_p, cats = func(X_tr.copy(), X_te.copy(), gen_feats.copy(), [])
        data[name] = (X_tr_p, X_te_p, gen_p, cats, y_tr, y_te, gen_names)  # Spit out the values.

    return data


# First, we define a function to accommodate all data (to provide category flag + fill-in missing entries) to objects and also split X, y into train and test

def cat_fill_split(df, new_df):
    """
    (OBS there should exist a function before this one to add those NEW FEATURES when engineering)

    This function will do three main things:

    1) Fill in missing values for Type 2.
    2) Check all objects in dataframes and assign category tag.
    3) Split between training and test values

    Args:
        df (DataFrame): the dataframe to train and test
        new_df (DataFrame): the new dataframe for new pokemon.

    Returns:
        X_train, X_test, y_train, y_test and all previous dataframe's objects conversed to categorical features.
    """

    # As the new_pokes dataframe will contain names, we need to drop, as this will not be required for the whole training.
    new_df = new_df.drop('name', axis=1)

    # We need to imput missing types for type_2 and transform some of them to categories in tdf. We also gonna transform objects into categories in tdf.
    to_cat = [df, new_df]
    for dataframe in to_cat:
        dataframe['type_2'] = dataframe['type_2'].fillna('None')
        dataframe[dataframe.select_dtypes('object').columns] = dataframe.select_dtypes('object').astype('category')

    X = df.drop('total_stats', axis=1)
    y = df['total_stats']

    # Defining the train/test splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

    return X_train, X_test, y_train, y_test, new_df

# We now define a preprocessor for catboost with ordinality transformation


def prep_catboost_ordinal(X_train,
                          X_test,
                          new_gen,
                          new_cat_feats=None,  # In case new categorical features appear
                          new_maps=None):  # To enhance if future feat. eng.
    """
    Preproccessor for catboostregressor. In this case, we create a map for two categories, to transform into simple numbers. 

    Args:
        X_train (DataFrame): 
        X_test (DataFrame): 
        new_gen (DataFrame): 
        new_cat_feats (list, optional): In case new categories appear in feat. eng.
        new_maps (dict, optional): In case previous new cats require a mapping, this is the place

    Returns:
        X_train, X_test, new_gen, cats. All preprocessed.
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

    dfs = [X_train, X_test, new_gen]
    dfs_out = []

    # In case of new maps
    if new_maps is not None:  # Must be a dict of dicts
        maps.update(new_maps)

    if new_cat_feats is not None:
        cat_feats.extend(new_cat_feats)

    for col, mapping in maps.items():
        for df in dfs:
            df[col] = df[col].map(mapping).astype('float')

    dfs_out.extend(dfs)

    return tuple(dfs_out) + (cat_feats,)


# Testing
# my_reg_cat = ['type_1', 'type_2', 'shape', 'color']
# X_tr, X_te, gen_cat_ord, _ = prep_catboost_ordinal(X_train, X_test, new_gen_initials,my_reg_cat)
# shape_checker(X_tr, X_te, gen_cat_ord)


# I will define now with catboost with natural, without any strange ordinality

def prep_catboost_native(X_train,
                         X_test,
                         new_gen,
                         new_cat_feats=None):  # To enhance if future feat. eng.
    """
    Another preprocessor function for catboost, but in this case native.

    Args:
        X_train (DataFrame): 
        X_test (DataFrame): 
        new_gen (DataFrame): 
        new_cat_feats (list, optional): If new categories appear after feat. eng. Defaults to None.

    Returns:
        X_train, X_test, new_gen, cats. All preprocessed.
    """

    cat_feats = ['type_1', 'type_2', 'rarity', 'stage', 'shape', 'color']

    if new_cat_feats is not None:
        cat_feats.extend(new_cat_feats) if isinstance(new_cat_feats, list) else [new_cat_feats]

    # Transforming those columns to category
    dfs = [X_train.copy(), X_test.copy(), new_gen.copy()]
    for df in dfs:
        for col in cat_feats:
            df[col] = df[col].astype('category')

    return X_train, X_test, new_gen, cat_feats


# Testing
# X_tr, X_te, gen_cat_nat, _ = prep_catboost_native(X_train, X_test, new_gen_initials)
# shape_checker(X_tr, X_te, gen_cat_nat)


# Finally, time to define same preprocessor but for lightgbm

def prep_lightgbm(X_train,
                  X_test,
                  new_gen,
                  new_cat_feats=None):  # To enhance if future feat. eng.
    """
    Another preprocessor function for LGBM.

    Args:
        X_train (DataFrame): 
        X_test (DataFrame): 
        new_gen (DataFrame): 
        new_cat_feats (list, optional): If new categories appear after feat. eng. Defaults to None.

    Returns:
        X_train, X_test, new_gen, cats. All preprocessed.
    """

    cat_feats = ['type_1', 'type_2', 'rarity', 'stage', 'shape', 'color']

    if new_cat_feats is not None:
        cat_feats.extend(new_cat_feats) if isinstance(new_cat_feats, list) else [new_cat_feats]

    # Transforming those columns to category
    dfs = [X_train, X_test, new_gen]
    for df in dfs:
        for col in cat_feats:
            df[col] = df[col].astype('category')

    return X_train, X_test, new_gen, cat_feats


def shape_checker(x_train, x_test, new_gen_df):
    print("Shapes:", x_train.shape, x_test.shape, new_gen_df.shape)
    print("Columns match:", (x_train.columns == x_test.columns).all())
    print("Indices preserved:", x_train.index.equals(x_train.index))
