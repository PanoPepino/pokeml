import pandas as pd
import numpy as np


from pokeml.utils.utils_eda import validate_required_columns


def extract_type_deviations(df):
    """
    This function will read a dataframe, group all pokes by stage, rarity and then check how pokemons of a given primary and secondary type deviate wrt the median of their stage.
    """
    required_cols = {"type_1", "type_2", "stage", "rarity", "total_stats"}
    validate_required_columns(df, required_cols, func_name="extract_type_deviations")

    # Grouping
    mono = df[df["type_2"].isna()].groupby(["type_1", "stage", "rarity"])["total_stats"].agg(
        mean="mean",
        median="median",
        count="count"
    )

    dual_t1 = df[df["type_2"].notna()].groupby(["type_1", "stage", "rarity"])["total_stats"].agg(
        mean="mean",
        median="median",
        count="count"
    )

    dual_t2 = df[df["type_2"].notna()].groupby(["type_2", "stage", "rarity"])["total_stats"].agg(
        mean="mean",
        median="median",
        count="count"
    )

    # Removing index
    flat_mono = mono.reset_index()
    flat_dual_t1 = dual_t1.reset_index()
    flat_dual_t2 = dual_t2.reset_index().rename(columns={"type_2": "type_1"})  # Rephrasing typing

    regular = df[df["rarity"].eq("regular")]
    legendary = df[df["rarity"].eq("legendary")]

    regular_baseline = regular.groupby("stage")["total_stats"].median()
    legendary_baseline = legendary["total_stats"].median()

    for sub in [flat_mono, flat_dual_t1, flat_dual_t2]:
        sub["deviation"] = np.nan

    # Checking devations for regular and legendary for mono and dual type pokes
    reg_mask = flat_mono["rarity"].eq("regular")
    leg_mask = flat_mono["rarity"].eq("legendary")
    flat_mono.loc[reg_mask, "deviation"] = (
        flat_mono.loc[reg_mask, "median"] - flat_mono.loc[reg_mask, "stage"].map(regular_baseline)
    )
    flat_mono.loc[leg_mask, "deviation"] = flat_mono.loc[leg_mask, "median"] - legendary_baseline
    flat_mono["construction"] = "mono"

    reg_mask = flat_dual_t1["rarity"].eq("regular")
    leg_mask = flat_dual_t1["rarity"].eq("legendary")
    flat_dual_t1.loc[reg_mask, "deviation"] = (
        flat_dual_t1.loc[reg_mask, "median"] - flat_dual_t1.loc[reg_mask, "stage"].map(regular_baseline)
    )
    flat_dual_t1.loc[leg_mask, "deviation"] = flat_dual_t1.loc[leg_mask, "median"] - legendary_baseline
    flat_dual_t1["construction"] = "dual_t1"

    reg_mask = flat_dual_t2["rarity"].eq("regular")
    leg_mask = flat_dual_t2["rarity"].eq("legendary")
    flat_dual_t2.loc[reg_mask, "deviation"] = (
        flat_dual_t2.loc[reg_mask, "median"] - flat_dual_t2.loc[reg_mask, "stage"].map(regular_baseline)
    )
    flat_dual_t2.loc[leg_mask, "deviation"] = flat_dual_t2.loc[leg_mask, "median"] - legendary_baseline
    flat_dual_t2["construction"] = "dual_t2"

    flat_all = pd.concat([flat_mono, flat_dual_t1, flat_dual_t2], ignore_index=True)
    flat_all = flat_all.rename(columns={"type_1": "type"})

    df_w_deviations = flat_all.reindex(columns=["rarity", "stage", "type", "construction",
                                                "count", "mean", "median", "deviation"])
    df_w_deviations["mean"] = df_w_deviations["mean"].round(1)

    return df_w_deviations, regular_baseline, legendary_baseline


def compare_type_ordering(stage, baseline, df, rarity="regular", min_count=3, use_zscore=True):
    """
    This function eats previous df_w_deviations and checks how bst of pokes at some stage and rarity with type_1 and type_2 differ wrt the baseline of their stage. In this way one extracts information on how typing may affect ordering.

    Args:
        stage (_type_): _description_
        baseline (_type_): _description_
        df (_type_): _description_
        rarity (str, optional): _description_. Defaults to "regular".
        min_count (int, optional): _description_. Defaults to 3.
        use_zscore (bool, optional): _description_. Defaults to True.

    Returns:
        comparison (DataFrame)
    """
    required_cols = {"rarity", "stage", "type", "construction", "count", "deviation"}
    validate_required_columns(df, required_cols, func_name="compare_type_ordering")

    if rarity == "legendary":
        subset = df[
            (df["rarity"] == "legendary") &
            (df["construction"].isin(["dual_t1", "dual_t2"]))
        ].copy()
    else:
        subset = df[
            (df["stage"] == stage) &
            (df["rarity"] == "regular") &
            (df["construction"].isin(["dual_t1", "dual_t2"]))
        ].copy()

    if subset.empty:
        return pd.DataFrame()

    pivot_dev = subset.pivot_table(
        index="type",
        columns="construction",
        values="deviation",
        aggfunc="first"
    )

    pivot_count = subset.pivot_table(
        index="type",
        columns="construction",
        values="count",
        aggfunc="first"
    ).fillna(0).astype(int)

    if not {"dual_t1", "dual_t2"}.issubset(pivot_dev.columns):
        return pd.DataFrame()
    if not {"dual_t1", "dual_t2"}.issubset(pivot_count.columns):
        return pd.DataFrame()

    comparison = pd.DataFrame({
        "type_2": pivot_dev.index,
        "dual_t1_dev": pivot_dev["dual_t1"],
        "dual_t2_dev": pivot_dev["dual_t2"],
        "n_t1": pivot_count["dual_t1"],
        "n_t2": pivot_count["dual_t2"]
    }).reset_index(drop=True)

    comparison = comparison[
        (comparison["n_t1"] >= min_count) &
        (comparison["n_t2"] >= min_count)
    ].copy()

    if comparison.empty:
        return comparison

    if use_zscore:
        all_vals = pd.concat(
            [comparison["dual_t1_dev"], comparison["dual_t2_dev"]],
            ignore_index=True
        ).dropna()

        mu = all_vals.mean()
        sigma = all_vals.std(ddof=0)

        if sigma == 0 or pd.isna(sigma):
            comparison["dev_z_t1"] = np.nan
            comparison["dev_z_t2"] = np.nan
        else:
            comparison["dev_z_t1"] = (comparison["dual_t1_dev"] - mu) / sigma
            comparison["dev_z_t2"] = (comparison["dual_t2_dev"] - mu) / sigma

        comparison["difference"] = comparison["dev_z_t1"] - comparison["dev_z_t2"]
    else:
        comparison["difference"] = comparison["dual_t1_dev"] - comparison["dual_t2_dev"]

    comparison["abs_diff"] = comparison["difference"].abs()
    comparison["baseline"] = baseline.loc[stage] if rarity == "regular" else float(baseline)

    return comparison.sort_values("abs_diff", ascending=False)


def fit_dual_type_dev_map(df, min_count=3, use_zscore=True):
    """_summary_

    Args:
        df (_type_): _description_
        min_count (int, optional): _description_. Defaults to 3.
        use_zscore (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    required_cols = {"rarity", "stage", "type", "construction", "count", "deviation"}
    validate_required_columns(df, required_cols, func_name="fit_dual_type_dev_map")

    subset = df[df["construction"].isin(["dual_t1", "dual_t2"])].copy()
    if subset.empty:
        return pd.DataFrame()

    pivot_dev = subset.pivot_table(
        index=["rarity", "stage", "type"],
        columns="construction",
        values="deviation",
        aggfunc="first",
    )

    pivot_count = subset.pivot_table(
        index=["rarity", "stage", "type"],
        columns="construction",
        values="count",
        aggfunc="first",
    ).fillna(0).astype(int)

    needed = {"dual_t1", "dual_t2"}
    if not needed.issubset(pivot_dev.columns) or not needed.issubset(pivot_count.columns):
        return pd.DataFrame()

    comparison = pd.DataFrame({
        "rarity": pivot_dev.index.get_level_values("rarity"),
        "stage": pivot_dev.index.get_level_values("stage"),
        "type": pivot_dev.index.get_level_values("type"),
        "dual_t1_dev": pivot_dev["dual_t1"].values,
        "dual_t2_dev": pivot_dev["dual_t2"].values,
        "n_t1": pivot_count["dual_t1"].values,
        "n_t2": pivot_count["dual_t2"].values,
    })

    comparison = comparison[
        (comparison["n_t1"] >= min_count) &
        (comparison["n_t2"] >= min_count)
    ].copy()

    if comparison.empty:
        return comparison

    if use_zscore:
        all_vals = pd.concat(
            [comparison["dual_t1_dev"], comparison["dual_t2_dev"]],
            ignore_index=True
        ).dropna()

        mu = all_vals.mean()
        sigma = all_vals.std(ddof=0)

        if pd.isna(sigma) or sigma == 0:
            comparison["dev_z_t1"] = np.nan
            comparison["dev_z_t2"] = np.nan
        else:
            comparison["dev_z_t1"] = (comparison["dual_t1_dev"] - mu) / sigma
            comparison["dev_z_t2"] = (comparison["dual_t2_dev"] - mu) / sigma
    else:
        comparison["dev_z_t1"] = comparison["dual_t1_dev"]
        comparison["dev_z_t2"] = comparison["dual_t2_dev"]

    comparison["dev_z_t1_isna"] = comparison["dev_z_t1"].isna().astype("int8")
    comparison["dev_z_t2_isna"] = comparison["dev_z_t2"].isna().astype("int8")

    df_comparison = comparison[[
        "rarity", "stage", "type",
        "dev_z_t1", "dev_z_t2",
        "dev_z_t1_isna", "dev_z_t2_isna"
    ]]

    return df_comparison


def fit_mono_type_dev_map(df, min_count=3, use_zscore=True):
    required_cols = {"type_1", "type_2", "rarity", "stage", "total_stats"}
    validate_required_columns(df, required_cols, func_name="fit_mono_type_dev_map")

    mono = df[df["type_2"].isna()].groupby(["rarity", "stage", "type_1"])["total_stats"].agg(
        mean="mean",
        median="median",
        count="count",
    ).reset_index()

    regular = df[df["rarity"].eq("regular")]
    legendary = df[df["rarity"].eq("legendary")]

    regular_baseline = regular.groupby("stage")["total_stats"].median()
    legendary_baseline = legendary["total_stats"].median()

    mono["deviation"] = np.nan
    reg_mask = mono["rarity"].eq("regular")
    leg_mask = mono["rarity"].eq("legendary")

    mono.loc[reg_mask, "deviation"] = (
        mono.loc[reg_mask, "median"] - mono.loc[reg_mask, "stage"].map(regular_baseline)
    )
    mono.loc[leg_mask, "deviation"] = mono.loc[leg_mask, "median"] - legendary_baseline

    mono = mono[mono["count"] >= min_count].copy()
    if mono.empty:
        return mono, regular_baseline, legendary_baseline

    if use_zscore:
        vals = mono["deviation"].dropna()
        mu = vals.mean()
        sigma = vals.std(ddof=0)

        if pd.isna(sigma) or sigma == 0:
            mono["mono_dev_z"] = np.nan
        else:
            mono["mono_dev_z"] = (mono["deviation"] - mu) / sigma
    else:
        mono["mono_dev_z"] = mono["deviation"]

    mono["mono_dev_z_isna"] = mono["mono_dev_z"].isna().astype("int8")

    df_mono = mono[[
        "rarity", "stage", "type_1",
        "mono_dev_z", "mono_dev_z_isna"
    ]]

    return df_mono, regular_baseline, legendary_baseline
