"""EDA population, interpretation and temporal leakage regression tests."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pokeml.eda.general import get_extremes, bst_histogram, split_bst_dist
from pokeml.eda.stages import stage_outliers, project_to_mock_stage
from pokeml.eda.power_creep import default_species, fixed_composition, drift_backtest
from pokeml.eda.types import extract_type_deviations, compare_type_ordering, compare_exact_type_ordering
from pokeml.visualisation.viz_eda.stage_buckets import plot_stage_buckets

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    return pd.read_csv(ROOT / 'datasets/pkdx_min.csv')


def test_extrema_keep_ties_and_empty_schema():
    frame = pd.DataFrame({'name': ['a', 'b', 'c'], 'generation': [2] * 3,
                          'total_stats': [100, 600, 600], 'rarity': ['regular'] * 3})
    result = get_extremes(frame, 'total_stats')
    assert result.loc[result.extreme.eq('max'), 'name'].tolist() == ['b', 'c']
    empty = get_extremes(frame, 'total_stats', generation=5)
    assert list(empty) == list(result)
    minimum, maximum, legends = split_bst_dist(frame)
    assert minimum == [100] and maximum == [600]
    assert np.isnan(legends[0])


def test_histogram_accounts_for_every_boundary_value():
    frame = pd.DataFrame({'total_stats': [160, 170, 180, 720, 740, 760]})
    table = bst_histogram(frame)
    assert table['count'].sum() == len(frame)
    assert table.lower.min() <= frame.total_stats.min()
    assert table.upper.max() > frame.total_stats.max()


def test_stage_outliers_and_shared_legend():
    frame = snapshot()
    outliers = stage_outliers(frame)
    final = outliers[outliers.stage.eq('s3c3') & outliers.side.eq('upper')]
    assert len(final) == 12
    assert set(final.total_stats) == {600, 670}
    assert set(final.upper_fence) == {585}
    fig, ax = plot_stage_buckets(frame)
    assert len(ax.get_legend().get_texts()) == 4
    plt.close(fig)


def test_original_mock_projection_is_stage_centring():
    frame = snapshot()
    result, medians, reference, _ = project_to_mock_stage(frame)
    regular = frame.rarity.eq('regular')
    np.testing.assert_allclose(result.loc[regular, 'bst_projected_mock_stage'],
                               frame.loc[regular, 'total_stats'] - frame.loc[regular, 'stage'].map(medians) + reference)
    assert 'bst_projected_mock_stage' not in frame


def test_fixed_composition_removes_roster_mix_only():
    records = []
    for generation, first, last in [(1, 9, 1), (2, 1, 9)]:
        for stage, count, value in [('s1c2', first, 100), ('s2c2', last, 500)]:
            records += [{'generation': generation, 'rarity': 'regular', 'stage': stage,
                         'total_stats': value}] * count
    comparison, weights = fixed_composition(pd.DataFrame(records), min_count=1)
    assert comparison.actual_mean.tolist() == [140, 460]
    np.testing.assert_allclose(comparison.fixed_composition_mean, [300, 300])
    assert weights.weight.sum() == 1
    assert weights.supported.all()


def test_drift_forecasts_do_not_use_last_holdout_values():
    frame = default_species(snapshot())
    first = drift_backtest(frame)
    frame.loc[frame.generation.eq(9), 'total_stats'] += 10000
    second = drift_backtest(frame)
    pd.testing.assert_series_equal(first.prediction, second.prediction)
    assert (first.train_last_generation < first.validation_generation).all()
    assert not first.loc[first.validation_generation.eq(9), 'mae'].equals(
        second.loc[second.validation_generation.eq(9), 'mae'])


def test_pooled_legendary_medians_use_actual_rows():
    frame = snapshot()
    groups, _, baseline = extract_type_deviations(frame)
    table = compare_type_ordering(None, baseline, groups, rarity='legendary', use_zscore=False)
    work = frame[frame.rarity.eq('legendary') & frame.type_2.notna()]
    expected = work.groupby('type_1').total_stats.median() - baseline
    for row in table.itertuples():
        assert row.dual_t1_dev == expected.loc[row.type_2]


def test_swapped_pairs_never_mix_unrelated_partners():
    frame = pd.DataFrame({'rarity': ['regular'] * 4, 'stage': ['s2c2'] * 4,
                          'type_1': ['fire', 'water', 'fire', 'grass'],
                          'type_2': ['water', 'fire', 'grass', 'poison'],
                          'total_stats': [400, 500, 600, 300]})
    table = compare_exact_type_ordering(frame)
    assert len(table) == 1
    assert table.pair.iloc[0] == 'fire / water'
    assert table.difference.iloc[0] == -100
    assert table['sparse'].iloc[0]


