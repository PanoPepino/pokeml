"""Descriptive stage buckets and morphology; these are not fitted ML features."""

from pokeml.data.validation import validate_required_columns

STAGE_ORDER = ('s1c3', 's1c2', 's2c3', 's2c2', 's3c3', 'single')


def stage_summary(df, min_count=10):
    """Counts, candidate medians and remaining variation by rarity and stage."""
    validate_required_columns(df, {'rarity', 'stage', 'total_stats'}, 'stage_summary')
    grouped = df.groupby(['rarity', 'stage'], observed=True).total_stats
    summary = grouped.agg(count='size', median='median', mean='mean', std='std', min='min', max='max')
    summary['q1'], summary['q3'] = grouped.quantile(.25), grouped.quantile(.75)
    summary['iqr'] = summary.q3 - summary.q1
    summary['sparse'] = summary['count'] < min_count
    return summary.reset_index()


def stage_outliers(df, rarity='regular'):
    """Named observations outside each stage's Tukey fences, with actual bounds."""
    summary = stage_summary(df)
    bounds = summary[summary.rarity.eq(rarity)][['stage', 'q1', 'q3', 'iqr']].copy()
    bounds['lower_fence'] = bounds.q1 - 1.5 * bounds.iqr
    bounds['upper_fence'] = bounds.q3 + 1.5 * bounds.iqr
    work = df[df.rarity.eq(rarity)].merge(bounds, on='stage', validate='many_to_one')
    work['side'] = work.total_stats.lt(work.lower_fence).map({True: 'lower', False: 'upper'})
    outside = work.total_stats.lt(work.lower_fence) | work.total_stats.gt(work.upper_fence)
    return work.loc[outside].sort_values(['stage', 'total_stats', 'name']).reset_index(drop=True)


def project_to_mock_stage(df):
    """Original dist_shape projection, retained as descriptive stage centring.

    Uses whole input population: never feed this fitted table into validation.
    Legendary rows retain their separate shared reference.
    """
    import numpy as np
    work = df.copy()
    regular_mask = work['rarity'].eq('regular')
    legendary_mask = work['rarity'].eq('legendary')

    # Stage medians for regular Pokémon
    stage_medians = work.loc[regular_mask].groupby('stage')['total_stats'].median()
    # Shared reference stage for regular Pokémon
    reference_regular = work.loc[regular_mask, 'total_stats'].median()
    # Legendary reference can be separate
    reference_legendary = work.loc[legendary_mask, 'total_stats'].median()
    work['stage_median'] = np.nan
    work['bst_projected_mock_stage'] = np.nan

    # Regular: project to the common mock stage
    work.loc[regular_mask, 'stage_median'] = work.loc[regular_mask, 'stage'].map(stage_medians)
    work.loc[regular_mask, 'bst_projected_mock_stage'] = (
        work.loc[regular_mask, 'total_stats'] - work.loc[regular_mask, 'stage_median'] + reference_regular
    )
    # Legendary: project to its own common baseline, if you want it included
    work.loc[legendary_mask, 'stage_median'] = reference_legendary
    work.loc[legendary_mask, 'bst_projected_mock_stage'] = (
        work.loc[legendary_mask, 'total_stats'] - reference_legendary + reference_legendary
    )
    return work, stage_medians, reference_regular, reference_legendary


def stage_residuals(df):
    """Whole-population descriptive BST minus median of rarity/stage bucket."""
    out = df.copy()
    out['stage_baseline'] = out.groupby(['rarity', 'stage'], observed=True).total_stats.transform('median')
    out['stage_residual'] = out.total_stats - out.stage_baseline
    return out


def morphology_summary(df, min_count=10):
    """Shape associations after stage centring; correlations are not causal."""
    work = stage_residuals(df[df.rarity.eq('regular')])
    shapes = work.groupby('shape', observed=True).agg(
        count=('name', 'size'), median_bst=('total_stats', 'median'),
        median_residual=('stage_residual', 'median'), mean_residual=('stage_residual', 'mean'))
    shapes['sparse'] = shapes['count'] < min_count
    return shapes.sort_values('median_residual')
