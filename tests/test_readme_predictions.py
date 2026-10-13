"""README table preserves distinct uncertainty meanings and model ranking."""
import pandas as pd
import pytest

from scripts.update_readme_predictions import prediction_table, update_readme


def test_prediction_table_uses_rank_and_distinct_uncertainty_formats():
    ranking = pd.DataFrame({'model': ['direct_light_gbm', 'direct_cat_native'], 'MAE': [28.27, 28.52]})
    records = []
    for name in ('browt', 'pombon', 'gecqua'):
        records += [{'model': 'direct_cat_native', 'name': name, 'pred_bst': 300., 'predictive_std': 20.},
                    {'model': 'direct_light_gbm', 'name': name, 'pred_bst': 310., 'lower_80': 315., 'upper_80': 355.}]
    table = prediction_table(pd.DataFrame(records), ranking)
    assert table.index('LightGBM') < table.index('CatBoost native')
    assert '310 [315, 355] †' in table
    assert '300 ± 20' in table
    assert '28.27' in table and '28.52' in table


def test_readme_update_only_changes_prediction_block(tmp_path):
    path = tmp_path / 'README.md'
    path.write_text('Keep introduction\n<!-- PREDICTIONS -->\nold\n<!-- /PREDICTIONS -->\nKeep diagram')
    update_readme(path, 'new table')
    assert path.read_text() == 'Keep introduction\n<!-- PREDICTIONS -->\n\nnew table\n\n<!-- /PREDICTIONS -->\nKeep diagram'
    path.write_text('No prediction markers')
    with pytest.raises(ValueError, match='marker'):
        update_readme(path, 'new table')
