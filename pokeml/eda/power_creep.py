"""Generation associations and forward tests of stage-baseline drift.

Current stats/chains are a retrospective snapshot, not historical stat records.
Nine generation aggregates cannot establish a causal design trend.
"""

import numpy as np
import pandas as pd

from pokeml.data.validation import validate_required_columns


def default_species(df):
    """One default record per species; selected alternate forms analysed separately."""
    validate_required_columns(df, {'id', 'species_id'}, 'default_species')
    out = df[df.id.eq(df.species_id)].copy()
    if out.species_id.duplicated().any():
        raise ValueError('Default species records must be unique')
    return out


def generation_summary(df):
    """Whole input roster medians, means, extrema and explicit sample sizes."""
    return df.groupby('generation', observed=True).agg(
        count=('name', 'size'), species=('species_id', 'nunique'),
        median=('total_stats', 'median'), mean=('total_stats', 'mean'),
        min=('total_stats', 'min'), max=('total_stats', 'max')).reset_index()


def fixed_composition(df, min_count=5):
    """Regular BST means at constant stage weights, not an average of medians.

    Pooled weights are descriptive and use full input. Only stages supported in
    every generation enter; return support/weights so exclusions stay visible.
    Never use this full-population table as a fitted forecasting feature.
    """
    work = df[df.rarity.eq('regular')]
    counts = pd.crosstab(work.generation, work.stage)
    supported = counts.ge(min_count).all(axis=0)
    stages = supported.index[supported].tolist()
    if not stages:
        raise ValueError('No stage supported in every generation')
    pooled = work.stage.value_counts().reindex(counts.columns, fill_value=0)
    weights = pooled.reindex(stages) / pooled.reindex(stages).sum()
    support = pd.DataFrame({'min_generation_count': counts.min(), 'supported': supported,
                            'weight': weights.reindex(counts.columns, fill_value=0)})
    means = work.groupby(['generation', 'stage'], observed=True).total_stats.mean().unstack()
    result = work.groupby('generation').total_stats.agg(actual_mean='mean', actual_median='median')
    result['fixed_composition_mean'] = means[stages].mul(weights).sum(axis=1)
    return result.reset_index(), support.reset_index(names='stage')


def polynomial_trend(df, stage, degree=1, start_generation=1):
    """Reproduce historical polynomial hypotheses; R² here is in-sample only."""
    work = df[df.rarity.eq('regular') & df.stage.eq(stage) & df.generation.ge(start_generation)]
    table = work.groupby('generation').total_stats.agg(count='size', mean='mean', median='median').reset_index()
    if degree < 0 or len(table) <= degree:
        raise ValueError('Polynomial needs more generation observations than degree')
    rows = []
    for statistic in ['mean', 'median']:
        fitted = np.polynomial.Polynomial.fit(table.generation, table[statistic], degree)
        predicted = fitted(table.generation)
        table[f'{statistic}_fit'] = predicted
        residual = np.square(table[statistic] - predicted).sum()
        total = np.square(table[statistic] - table[statistic].mean()).sum()
        rows.append({'stage': stage, 'statistic': statistic, 'degree': degree,
                     'generations': len(table), 'in_sample_r2': 1 - residual / total if total else np.nan})
    return pd.DataFrame(rows), table


def drift_backtest(df, validation_generations=(6, 7, 8, 9), stages=('s3c3', 'single')):
    """Compare candidate median baselines on strictly later generations.

    Frozen median, recent-three-generation median, linear generation-median
    trend, and original hypotheses: cubic F3 / linear singles since Gen 3.
    No clipping, target-derived archetype flags or held-out fitting.
    """
    work = df[df.rarity.eq('regular')]
    rows = []
    for generation in validation_generations:
        for stage in stages:
            train = work[work.generation.lt(generation) & work.stage.eq(stage)]
            valid = work[work.generation.eq(generation) & work.stage.eq(stage)]
            if train.empty or valid.empty:
                continue
            medians = train.groupby('generation').total_stats.median()
            predictions = {
                'frozen_median': train.total_stats.median(),
                'recent_3_median': train.loc[train.generation.isin(medians.index[-3:]), 'total_stats'].median(),
            }
            if len(medians) >= 2:
                predictions['linear_drift'] = float(np.polynomial.Polynomial.fit(medians.index, medians, 1)(generation))
            degree, start = (3, 1) if stage == 's3c3' else (1, 3)
            original = medians[medians.index >= start]
            if len(original) > degree:
                predictions['original_polynomial'] = float(np.polynomial.Polynomial.fit(original.index, original, degree)(generation))
            for method, prediction in predictions.items():
                errors = prediction - valid.total_stats.to_numpy()
                rows.append({'validation_generation': generation, 'stage': stage, 'method': method,
                             'train_last_generation': int(train.generation.max()), 'train_count': len(train),
                             'validation_count': len(valid), 'prediction': prediction,
                             'mae': np.abs(errors).mean(), 'rmse': np.sqrt(np.square(errors).mean()),
                             'signed_error': errors.mean()})
    return pd.DataFrame(rows)
