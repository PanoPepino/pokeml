"""Retired CLI boundary and explicit historical replay without implicit fitting."""
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
from typer.testing import CliRunner

from pokeml.cli import app
from pokeml.models.deployment import PokemonPredictor
from pokeml.models.residual import BaselineCorrectionRegressor
from pokeml.pipeline.workflow import load_historical_workflow_result, load_workflow_result

PARENT = Path('artifacts/workflow/pokemon_tuned_60_v1')


def test_cli_exposes_current_commands_and_rejects_retired_routes():
    runner = CliRunner()
    current = runner.invoke(app, ['model', '--help'])
    assert current.exit_code == 0
    for command in ('workflow', 'predict-release', 'training-curves'):
        assert command in current.output
    for command in ('tune', 'train', 'predict'):
        result = runner.invoke(app, ['model', command])
        assert result.exit_code != 0
        assert 'No such command' in result.output
    assert runner.invoke(app, ['evaluation', '--help']).exit_code != 0


def test_explicit_historical_tables_and_release_predict_without_fit(monkeypatch):
    if not PARENT.exists():
        pytest.skip('Local historical artifacts are not part of package distribution')
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical replay and inference must not fit')
    monkeypatch.setattr(BaselineCorrectionRegressor, 'fit', forbidden)
    result = load_historical_workflow_result(PARENT)
    assert result.selected_model == 'direct_cat_native'
    with pytest.raises(ValueError, match='source changed'):
        load_workflow_result(PARENT)
    predictions = PokemonPredictor.load(result.release_directory).predict(pd.read_csv('datasets/new_pokes.csv'))
    expected = pd.read_csv(PARENT / 'predictions_details.csv')
    pd.testing.assert_series_equal(predictions.pred_bst, expected.pred_bst)


def test_historical_replay_rejects_incomplete_or_changed_data(tmp_path):
    if not PARENT.exists():
        pytest.skip('Local historical artifacts are not part of package distribution')
    source = next(PARENT.glob('*_model_manifest.json'))
    metadata = json.loads(source.read_text())
    target = tmp_path / source.name
    metadata['status'] = 'failed'
    target.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='incomplete'):
        load_historical_workflow_result(tmp_path)
    metadata['status'] = 'complete'
    changed = tmp_path / 'training.csv'
    shutil.copy(metadata['training_csv'], changed)
    changed.write_bytes(changed.read_bytes() + b'\n')
    metadata['training_csv'] = str(changed)
    target.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='training dataset changed'):
        load_historical_workflow_result(tmp_path)


