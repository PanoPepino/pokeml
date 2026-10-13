import pandas as pd
import numpy as np


from pokeml.data.validation import validate_required_columns


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

    # Preserve exact pooled legendary medians. Group medians cannot be pooled
    # by picking the first stage (the former ordering implementation did so).
    pooled = []
    for construction, column, mask in [
        ('mono', 'type_1', legendary.type_2.isna()),
        ('dual_t1', 'type_1', legendary.type_2.notna()),
        ('dual_t2', 'type_2', legendary.type_2.notna()),
    ]:
        table = legendary.loc[mask].groupby(column).total_stats.agg(count='size', mean='mean', median='median').reset_index()
        table = table.rename(columns={column: 'type'})
        table['rarity'], table['stage'], table['construction'] = 'legendary', 'all', construction
        table['deviation'] = table['median'] - legendary_baseline
        pooled.extend(table.to_dict('records'))
    df_w_deviations.attrs['pooled_legendary'] = pooled

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
        if stage is None and df.attrs.get('pooled_legendary'):
            df = pd.DataFrame(df.attrs['pooled_legendary'])
        elif stage is None and df.loc[df.rarity.eq('legendary'), 'stage'].nunique() > 1:
            raise ValueError('Pooled legendary comparison needs extract_type_deviations output with raw-derived pooled medians')
        subset = df[
            (df["rarity"] == "legendary") &
            (df["construction"].isin(["dual_t1", "dual_t2"]))
        ].copy()
        if stage is not None:
            subset = subset[subset.stage.eq(stage)]
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


def type_presence_summary(df, dual_only=False, min_count=10):
    """Presence associations, counting a Pokémon once per type, not per slot."""
    from pokeml.eda.stages import stage_residuals
    work = stage_residuals(df)
    if dual_only:
        work = work[work.type_2.notna()]
    work = work.assign(type=[list(dict.fromkeys([a, *([] if pd.isna(b) else [b])]))
                            for a, b in zip(work.type_1, work.type_2)]).explode('type')
    grouped = work.groupby(['rarity', 'stage', 'type'], observed=True)
    table = grouped.agg(count=('name', 'size'), families=('evolution_chain_id', 'nunique'),
                        median_bst=('total_stats', 'median'), median_residual=('stage_residual', 'median'))
    table['sparse'] = table['count'] < min_count
    return table.reset_index()


def type_pair_summary(df, min_count=10):
    """Unordered partner combinations, separate from primary/secondary slots."""
    from pokeml.eda.stages import stage_residuals
    work = stage_residuals(df)
    work = work[work.type_2.notna()].copy()
    work['pair'] = [' / '.join(sorted((a, b))) for a, b in zip(work.type_1, work.type_2)]
    table = work.groupby(['rarity', 'stage', 'pair'], observed=True).agg(
        count=('name', 'size'), families=('evolution_chain_id', 'nunique'),
        median_bst=('total_stats', 'median'), median_residual=('stage_residual', 'median'))
    table['sparse'] = table['count'] < min_count
    return table.reset_index()


def compare_exact_type_ordering(df, min_count=10):
    """Compare A/B with B/A within identical stage and rarity; association only."""
    work = df[df.type_2.notna()].copy()
    work['pair'] = [' / '.join(sorted((a, b))) for a, b in zip(work.type_1, work.type_2)]
    records = []
    for (rarity, stage, pair), group in work.groupby(['rarity', 'stage', 'pair'], observed=True):
        a, b = pair.split(' / ')
        forward, reverse = group[group.type_1.eq(a)], group[group.type_1.eq(b)]
        if forward.empty or reverse.empty:
            continue
        records.append({'rarity': rarity, 'stage': stage, 'pair': pair,
                        'forward': f'{a}/{b}', 'reverse': f'{b}/{a}',
                        'n_forward': len(forward), 'n_reverse': len(reverse),
                        'forward_median': forward.total_stats.median(),
                        'reverse_median': reverse.total_stats.median(),
                        'difference': forward.total_stats.median() - reverse.total_stats.median(),
                        'sparse': min(len(forward), len(reverse)) < min_count})
    return pd.DataFrame(records, columns=['rarity', 'stage', 'pair', 'forward', 'reverse',
                                         'n_forward', 'n_reverse', 'forward_median',
                                         'reverse_median', 'difference', 'sparse'])


def bug_pair_analysis(df, min_count=10):
    """Dual Bug vs other dual types: stage, generation, and partner breakdowns.

    Non-Bug controls enter each of their two partner strata. Never pool partner
    rows as independent observations; underlying IDs and families overlap.
    """
    from pokeml.eda.stages import stage_residuals
    work = stage_residuals(df[df.rarity.eq('regular')])
    work = work[work.type_2.notna()].copy()
    work['bug'] = work.type_1.eq('bug') | work.type_2.eq('bug')
    def summarise(frame, keys):
        table = frame.groupby(keys, observed=True).agg(
            count=('name', 'size'), families=('evolution_chain_id', 'nunique'),
            median_bst=('total_stats', 'median'), median_residual=('stage_residual', 'median'))
        table['sparse'] = table['count'] < min_count
        return table.reset_index()
    work['partner'] = [[b if a == 'bug' else a] if bug else [a, b]
                       for a, b, bug in zip(work.type_1, work.type_2, work.bug)]
    return {'stage': summarise(work, ['stage', 'bug']),
            'generation': summarise(work, ['stage', 'generation', 'bug']),
            'partner': summarise(work.explode('partner'), ['stage', 'partner', 'bug'])}
