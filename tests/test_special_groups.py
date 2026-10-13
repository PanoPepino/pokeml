"""Design-group scoring, chronological search and selection policy."""
import json

import numpy as np
import pandas as pd
import pytest

from pokeml.evaluation.groups import (design_groups, grouped_metrics, pooled_group_summary,
                                      selection_mask, ULTRA_BEAST_IDS, PARADOX_IDS)


def test_snapshot_membership_and_counts():
    rows = pd.read_csv('datasets/pkdx_min.csv')
    groups = design_groups(rows)
    assert groups.eq('ultra_beast').sum() == 11
    assert groups.eq('paradox').sum() == 20
    assert set(rows.loc[groups.eq('ultra_beast'), 'species_id']) == ULTRA_BEAST_IDS
    assert set(rows.loc[groups.eq('paradox'), 'species_id']) == PARADOX_IDS
    for name in ('poipole', 'naganadel'):
        assert groups.loc[rows.name.eq(name)].iloc[0] == 'ultra_beast'
    assert selection_mask(rows).loc[rows.rarity.eq('legendary')].any()
    assert [int((selection_mask(rows) & rows.generation.eq(g)).sum()) for g in (6, 7, 8, 9)] == [72, 95, 133, 102]
    pd.testing.assert_series_equal(groups, design_groups(rows.drop(columns='species_id')))


def test_group_scores_pool_counts_and_leave_predictions_untouched():
    rows = pd.DataFrame(dict(name=['ordinary', 'nihilego', 'ordinary', 'great_tusk'],
                            generation=[6, 7, 9, 9], population=['selected_forms'] * 4,
                            model=['test'] * 4, actual_bst=[100.] * 4,
                            pred_bst=[99., 90., 97., 80.]))
    before = rows.copy(deep=True)
    metrics = grouped_metrics(rows)
    summary = pooled_group_summary(metrics).set_index('evaluation_scope')
    assert summary.loc['all', 'MAE'] == 8.5
    assert summary.loc['ordinary', 'MAE'] == 2
    assert summary.loc['ordinary', 'n_validation'] == 2
    assert summary.loc['all', 'RMSE'] == pytest.approx(np.sqrt((1+100+9+400)/4))
    assert summary.loc['all', 'bias'] == 8.5
    empty = metrics.query("generation==6 and evaluation_scope=='paradox'").iloc[0]
    assert empty.n_validation == 0 and np.isnan(empty.MAE) and np.isnan(empty.RMSE)
    pd.testing.assert_frame_equal(rows, before)


def test_search_filters_checks_but_retains_training_rows(tmp_path, monkeypatch):
    import pokeml.evaluation.tuning as module
    from sklearn.model_selection import RandomizedSearchCV as RealSearch
    rows = pd.read_csv('datasets/pkdx_min.csv').iloc[:80].copy()
    rows['generation'] = np.repeat([5, 6, 7, 8, 9], [20, 5, 10, 15, 30])
    rows.loc[rows.index[25], 'species_id'] = 793  # Ultra Beast becomes later training data.
    rows.loc[rows.index[50], 'species_id'] = 984
    observed = []
    def spy(*args, **kwargs):
        observed.append(kwargs['cv'])
        return RealSearch(*args, **kwargs)
    monkeypatch.setattr(module, 'RandomizedSearchCV', spy)
    grid = tmp_path / 'grid.json'
    grid.write_text(json.dumps({'common_params': {}, 'params_grid': {'xgboost': {'n_estimators': [2]}}}))
    module.run_search(rows, {'xgboost': {'n_estimators': 2}}, [], grid, 1,
                      selection_scope='ordinary', output_directory=tmp_path / 'ordinary')
    for splits in observed:
        assert [len(v) for _, v in splits] == [5, 9, 15, 29]
        assert 25 in splits[2][0] and 25 in splits[3][0]
        assert not any(25 in v or 50 in v for _, v in splits)
    observed.clear()
    module.run_search(rows, {'xgboost': {'n_estimators': 2}}, [], grid, 1, selection_scope='all')
    assert [len(v) for _, v in observed[0]] == [5, 10, 15, 30]
    rows.loc[rows.generation.eq(7), 'species_id'] = 793
    with pytest.raises(ValueError, match='empty'):
        module.run_search(rows, {'xgboost': {'n_estimators': 2}}, [], grid, 1,
                          selection_scope='ordinary')


def test_workflow_ranks_requested_group_and_refits_all_rows(tmp_path, monkeypatch):
    import pokeml.pipeline.workflow as module
    from pokeml.models.deployment import PokemonPredictor
    rows = pd.read_csv('datasets/pkdx_min.csv').iloc[:80].copy()
    rows['generation'] = np.repeat([5, 6, 7, 8, 9], [20, 5, 10, 15, 30])
    rows.loc[rows.index[25], 'species_id'] = 793
    rows.loc[rows.index[50], 'species_id'] = 984
    training = tmp_path / 'train.csv'
    rows.to_csv(training, index=False)
    original = module.evaluate_cohort
    def controlled(*args, **kwargs):
        metric, predictions = original(*args, **kwargs)
        name = args[3]
        ordinary = design_groups(predictions).eq('ordinary').to_numpy()
        error = (np.full(len(predictions), 100.) if name == 'stage_median'
                 else np.where(ordinary, 1., 2000.) if name == 'direct_xgboost'
                 else np.where(ordinary, 5., 1.))
        predictions['pred_bst'] = predictions.actual_bst - error
        predictions['error_bst'] = error
        predictions['absolute_error'] = error
        metric.update(MAE=float(abs(error).mean()), RMSE=float(np.sqrt((error**2).mean())))
        return metric, predictions
    monkeypatch.setattr(module, 'evaluate_cohort', controlled)
    for scope, expected in [('ordinary', 'direct_xgboost'), ('all', 'correction_xgboost')]:
        result = module.run_workflow(module.WorkflowConfig(
            training_csv=str(training), output_root=str(tmp_path), run_name=scope,
            parameters={'xgboost': {'n_estimators': 2}}, sensitivity=False,
            training_curves=False, selection_scope=scope))
        assert result.selected_model == expected
        release = PokemonPredictor.load(result.release_directory)
        assert release.metadata['n_train'] == 80
        assert release.metadata['selection_scope'] == scope
        assert result.development_summary.loc[result.development_summary.model.eq(expected), 'n_validation'].iloc[0] == 60
        replay = module.load_workflow_result(result.run_directory)
        pd.testing.assert_frame_equal(result.evaluation_summary, replay.evaluation_summary)
