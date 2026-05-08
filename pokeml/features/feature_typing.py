from pokeml.utils.utils_eda import validate_required_columns
from pokeml.data.eda_types import extract_type_deviations, fit_dual_type_dev_map, fit_mono_type_dev_map

import numpy as np


def add_dual_type_dev_features(df, dual_map):
    required_cols = {"rarity", "stage", "type_1", "type_2"}
    validate_required_columns(df, required_cols, func_name="add_dual_type_dev_features")

    df = df.copy()
    df["dev_z_t1"] = np.nan
    df["dev_z_t2"] = np.nan
    df["dev_z_t1_isna"] = 1
    df["dev_z_t2_isna"] = 1

    if dual_map is None or dual_map.empty:
        return df

    left = dual_map.rename(columns={"type": "type_1"})[
        ["rarity", "stage", "type_1", "dev_z_t1", "dev_z_t1_isna"]
    ].drop_duplicates(["rarity", "stage", "type_1"])

    right = dual_map.rename(columns={"type": "type_2"})[
        ["rarity", "stage", "type_2", "dev_z_t2", "dev_z_t2_isna"]
    ].drop_duplicates(["rarity", "stage", "type_2"])

    tmp = df.loc[df["type_2"].notna(), ["rarity", "stage", "type_1", "type_2"]].copy()
    if tmp.empty:
        return df

    tmp = tmp.merge(left, on=["rarity", "stage", "type_1"], how="left")
    tmp = tmp.merge(right, on=["rarity", "stage", "type_2"], how="left")

    df.loc[tmp.index, "dev_z_t1"] = tmp["dev_z_t1"].values
    df.loc[tmp.index, "dev_z_t2"] = tmp["dev_z_t2"].values
    df.loc[tmp.index, "dev_z_t1_isna"] = tmp["dev_z_t1_isna"].fillna(1).astype("int8").values
    df.loc[tmp.index, "dev_z_t2_isna"] = tmp["dev_z_t2_isna"].fillna(1).astype("int8").values

    return df


def add_mono_type_dev_features(df, mono_map):
    required_cols = {"rarity", "stage", "type_1", "type_2"}
    validate_required_columns(df, required_cols, func_name="add_mono_type_dev_features")

    df = df.copy()

    if mono_map is None or mono_map.empty:
        df["mono_dev_z"] = np.nan
        df["mono_dev_z_isna"] = 1
        return df

    mono = mono_map[["rarity", "stage", "type_1", "mono_dev_z", "mono_dev_z_isna"]].copy()
    df = df.merge(mono, on=["rarity", "stage", "type_1"], how="left")

    df.loc[df["type_2"].notna(), "mono_dev_z"] = np.nan
    df.loc[df["type_2"].notna(), "mono_dev_z_isna"] = 1

    df["mono_dev_z_isna"] = df["mono_dev_z_isna"].fillna(1).astype("int8")
    return df


def build_type_deviation_features(df, min_count=3, use_zscore=True):
    type_devs, regular_baseline, legendary_baseline = extract_type_deviations(df)

    dual_map = fit_dual_type_dev_map(type_devs, min_count=min_count, use_zscore=use_zscore)
    mono_map, _, _ = fit_mono_type_dev_map(df, min_count=min_count, use_zscore=use_zscore)

    out = add_dual_type_dev_features(df, dual_map)
    out = add_mono_type_dev_features(out, mono_map)

    return out, dual_map, mono_map, regular_baseline, legendary_baseline


def get_type_deviation_state(df, min_count=3, use_zscore=True):
    type_devs, regular_baseline, legendary_baseline = extract_type_deviations(df)

    dual_map = fit_dual_type_dev_map(
        type_devs,
        min_count=min_count,
        use_zscore=use_zscore
    )
    mono_map, _, _ = fit_mono_type_dev_map(
        df,
        min_count=min_count,
        use_zscore=use_zscore
    )

    return {
        "dual_map": dual_map,
        "mono_map": mono_map,
        "regular_baseline": regular_baseline,
        "legendary_baseline": legendary_baseline,
        "new_cat_feats": [],
        "new_maps": {}
    }


def apply_type_deviation_features(df, state):
    out = add_dual_type_dev_features(df, state["dual_map"])
    out = add_mono_type_dev_features(out, state["mono_map"])

    return out
