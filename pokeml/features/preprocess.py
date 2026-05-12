# Here, I will define 3 different preprocessors to adapt data before model training.
# For convenience, I have also defined a function to check shapes match after preprocessing.


# We define a preprocessor for catboost with ordinality transformation
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
