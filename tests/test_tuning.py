"""Search schedules, real parameter spaces, pooling and uncertainty lifecycle."""
import json

import numpy as np
import pandas as pd
import pytest

from pokeml.config import load_tuning_grid
from pokeml.evaluation.tuning import canonical_parameters, sample_configurations, trial_table
from pokeml.evaluation.forward import tune_candidates
from pokeml.models.factory import make_estimator
from pokeml.models.residual import BaselineCorrectionRegressor
from pokeml.pipeline.workflow import default_parameters


def test_reproducible_unique_defaults_and_pair_schedules():
    grids = load_tuning_grid('configs/tuning_forward.json')
    for variant, defaults in default_parameters().items():
        section = grids['catboost' if variant.startswith('cat_') else variant]
        first = sample_configurations(variant, defaults, section, 60)
        second = sample_configurations(variant, defaults, section, 60)
        assert first == second
        assert len(first) == 60
        assert first[0] == defaults
        assert len({json.dumps(config, sort_keys=True) for config in first}) == 60
        assert first != sample_configurations(variant, defaults, section, 60, seed=2)
        if variant == 'light_gbm':
            assert all(config['subsample_freq'] == 1 for config in first[1:])
        if variant.startswith('cat_'):
            assert all('n_estimators' not in config and 'max_depth' not in config for config in first)


@pytest.mark.parametrize('variant', ['cat_native', 'cat_ordinal', 'light_gbm', 'xgboost'])
def test_real_space_fits_with_canonical_parameters(variant):
    grids = load_tuning_grid('configs/tuning_forward.json')
    section = grids['catboost' if variant.startswith('cat_') else variant]
    options = sample_configurations(variant, default_parameters()[variant], section, 2)[1]
    options['iterations' if variant.startswith('cat_') else 'n_estimators'] = 5
    if variant in ('light_gbm', 'xgboost'):
        options['fit_uncertainty'] = False
    else:
        options['allow_writing_files'] = False
    rows = pd.read_csv('datasets/pkdx_min.csv').iloc[:60]
    model = BaselineCorrectionRegressor(variant, make_estimator(variant, options)).fit(rows, rows.total_stats)
    assert np.isfinite(model.predict(rows)).all()
    if variant == 'light_gbm':
        assert model.model_.quantile_models == {}
        assert model.model_.median_model.get_params()['subsample_freq'] == 1
    elif variant == 'xgboost':
        assert model.model_.quantile_models_ == {}


def test_conflicting_aliases_rejected():
    with pytest.raises(ValueError, match='choose'):
        canonical_parameters('cat_native', {'iterations': 5, 'n_estimators': 6})
    with pytest.raises(ValueError, match='choose'):
        canonical_parameters('cat_native', {'depth': 3, 'max_depth': 4})
    with pytest.raises(ValueError, match='boosting'):
        canonical_parameters('xgboost', {'early_stopping_rounds': 5})


def test_trial_pooling_and_tie_breaks():
    configs = [{'n_estimators': 5}, {'n_estimators': 10}]
    results = {'params': [{'estimator__n_estimators': 10}, {'estimator__n_estimators': 5}],
               'mean_fit_time': [1, 2], 'mean_score_time': [0.1, 0.2]}
    counts = [1, 2, 3, 10]
    abs_errors = [np.arange(n, dtype=float) + 1 for n in counts]
    for i, errors in enumerate(abs_errors):
        results[f'split{i}_test_MAE'] = [-errors.mean(), -errors.mean()]
        results[f'split{i}_test_RMSE'] = [-np.sqrt((errors ** 2).mean()), -np.sqrt((errors ** 2).mean())]
    table = trial_table(results, configs, counts, 'direct_xgboost')
    errors = np.concatenate(abs_errors)
    np.testing.assert_allclose(table.pooled_MAE, np.abs(errors).mean())
    np.testing.assert_allclose(table.pooled_RMSE, np.sqrt((errors ** 2).mean()))
    assert table.loc[table['rank'] == 1, 'trial_id'].iloc[0] == 0
    assert table.loc[table.is_default, 'trial_id'].iloc[0] == 0


def test_search_records_and_restores_uncertainty(tmp_path):
    rows = pd.read_csv('datasets/pkdx_min.csv').iloc[:80].copy()
    rows['generation'] = np.repeat([5, 6, 7, 8, 9], [20, 5, 10, 15, 30])
    grid = tmp_path / 'grid.json'
    grid.write_text(json.dumps({'common_params': {}, 'params_grid': {
        'xgboost': {'n_estimators': [4, 5], 'max_depth': [2, 3]}}}))
    directory = tmp_path / 'tuning'
    models, summary = tune_candidates(rows, {'xgboost': {'n_estimators': 3}}, [], grid, 2,
                                     output_directory=directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert set(manifest['completed_models']) == {'direct_xgboost', 'correction_xgboost'}
    trials = pd.read_csv(directory / 'trials.csv')
    assert len(trials) == 4
    direct = trials.query("model == 'direct_xgboost'").sort_values('trial_id')
    corrected = trials.query("model == 'correction_xgboost'").sort_values('trial_id')
    assert list(direct.params) == list(corrected.params)
    assert trials.groupby('model').is_default.sum().eq(1).all()
    for name in ('direct_xgboost', 'correction_xgboost'):
        assert not hasattr(models[name], 'baseline_')
        assert models[name].estimator.get_params()['fit_uncertainty'] is True
        models[name].fit(rows, rows.total_stats)
        assert 'lower_80' in models[name].predict_distribution(rows)
    assert len(summary) == 2


@pytest.mark.parametrize('variant', ['light_gbm', 'xgboost'])
def test_point_only_predictions_match_full_model(variant):
    rows = pd.read_csv('datasets/pkdx_min.csv').iloc[:60]
    point = BaselineCorrectionRegressor(variant, make_estimator(variant, {'n_estimators': 5, 'fit_uncertainty': False}))
    full = BaselineCorrectionRegressor(variant, make_estimator(variant, {'n_estimators': 5}))
    point.fit(rows, rows.total_stats)
    full.fit(rows, rows.total_stats)
    np.testing.assert_allclose(point.predict(rows), full.predict(rows))
    assert 'lower_80' in full.predict_distribution(rows)
    with pytest.raises(ValueError):
        point.predict_distribution(rows)


def test_failure_preserves_completed_candidate_checkpoint(tmp_path, monkeypatch):
    import pokeml.evaluation.tuning as module
    rows = pd.read_csv('datasets/pkdx_min.csv').iloc[:40].copy()
    rows['generation'] = np.repeat([5, 6, 7, 8, 9], 8)
    grid = tmp_path / 'grid.json'
    grid.write_text(json.dumps({'common_params': {}, 'params_grid': {
        'xgboost': {'n_estimators': [3], 'max_depth': [2]}}}))
    original_fit = module.RandomizedSearchCV.fit
    calls = 0
    def failing_second_fit(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interruption')
        return original_fit(self, *args, **kwargs)
    monkeypatch.setattr(module.RandomizedSearchCV, 'fit', failing_second_fit)
    directory = tmp_path / 'tuning'
    with pytest.raises(RuntimeError, match='interruption'):
        tune_candidates(rows, {'xgboost': {'n_estimators': 3}}, [], grid, 1,
                        output_directory=directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'running'
    assert manifest['completed_models'] == ['direct_xgboost']
    assert len(pd.read_csv(directory / 'trials.csv')) == 1
    assert not (directory / 'correction_xgboost_trials.csv').exists()
