"""XGBoost encoding, compatibility and orchestration contracts."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from pokeml.artifacts import MODEL_NAMES, tuning_section
from pokeml.models.estimators import XGB_Trainer
from pokeml.pipeline.prepare import prepare_data_split
from pokeml.pipeline.workflow import default_parameters
from pokeml.evaluation.forward import candidates


def test_unseen_categories_replay_encoding_and_feature_importance(tmp_path):
    X = pd.DataFrame({'kind': ['bug', 'water'] * 20, 'size': np.arange(40, dtype=float)})
    y = np.arange(40, dtype=float) * 8 + 200
    model = XGB_Trainer(n_estimators=12, max_depth=2).fit(X, y, eval_set=[(X, y), (X, y)])
    categories = model.encoder_.named_transformers_['categorical'].categories_[0].copy()
    query = pd.DataFrame({'kind': ['new', 'water'], 'size': [5., 8.]}, index=[7, 7])
    distribution = model.predict_distribution(query)
    assert all(len(values) == 2 for values in distribution.values())
    assert np.isfinite(distribution['pred_bst']).all()
    np.testing.assert_array_equal(categories, model.encoder_.named_transformers_['categorical'].categories_[0])
    assert 'validation_1' in model.get_evals()
    names, importance = model.get_top_features(k=2)
    assert set(names) == set(X.columns)
    np.testing.assert_allclose(sum(importance), 100)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(model.predict(query), joblib.load(path).predict(query))
    assert clone(model).get_params() == model.get_params()


def test_training_categories_exclude_validation_only_values():
    frame = pd.read_csv('datasets/pkdx_min.csv').iloc[:80]
    query = frame.iloc[60:].copy().assign(color='validation_only')
    prepared, _ = prepare_data_split(frame.iloc[:60], query)
    X, valid, target, _, _ = prepared['xgboost']
    model = XGB_Trainer(n_estimators=5).fit(X, target)
    assert 'validation_only' not in model.encoder_.named_transformers_['categorical'].categories_[model.cat_cols_.index('color')]
    assert np.isfinite(model.predict(valid)).all()


def test_all_four_variants_and_nine_candidates_registered():
    assert set(MODEL_NAMES) == {'xgboost', 'cat_native', 'cat_ordinal', 'light_gbm'}
    assert tuning_section('xgboost') == 'xgboost'
    assert len(candidates(default_parameters(), [])) == 9


def test_xgboost_prepared_prediction_cli_contract(tmp_path):
    from pokeml.cli import app
    from typer.testing import CliRunner
    frame = pd.read_csv('datasets/pkdx_min.csv').iloc[:80]
    from pokeml.models.residual import BaselineCorrectionRegressor
    from pokeml.models.deployment import PokemonPredictor
    composite = BaselineCorrectionRegressor('xgboost', XGB_Trainer(n_estimators=5))
    composite.fit(frame.iloc[:60], frame.iloc[:60].total_stats)
    release = tmp_path / 'release'
    PokemonPredictor(composite, 'xgboost/test').save(release)
    input_csv = tmp_path / 'announcements.csv'
    output_csv = tmp_path / 'release_predictions.csv'
    frame.iloc[60:].to_csv(input_csv, index=False)
    command = CliRunner().invoke(app, ['model', 'predict-release', '--release-directory',
        str(release), '--input-path', str(input_csv), '--output-path', str(output_csv)])
    assert command.exit_code == 0, command.output
    assert len(pd.read_csv(output_csv)) == 20
