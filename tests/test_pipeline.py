# scripts/smoke_test_pipeline.py

import pandas as pd
import pytest
from pathlib import Path
pytest.importorskip("sklearn")
pytest.importorskip("catboost")
pytest.importorskip("lightgbm")

from pokeml.pipeline.prepare import prepare_data_train


def test_validation_targets_cannot_change_training_state():
    from pokeml.features.feature_registry import get_feature_steps
    from pokeml.pipeline.prepare import prepare_data_split
    frame = pd.read_csv(Path(__file__).resolve().parents[1] / "datasets/pkdx_min.csv")
    train_frame, valid_frame = frame.iloc[:750], frame.iloc[750:].copy()
    steps = get_feature_steps("full")
    first, first_state = prepare_data_split(train_frame, valid_frame, steps)
    valid_frame["total_stats"] += 5000
    second, second_state = prepare_data_split(train_frame, valid_frame, steps)
    for name in first:
        pd.testing.assert_frame_equal(first[name][0], second[name][0])
        pd.testing.assert_frame_equal(first[name][1], second[name][1])
    for key in ("dual_map", "mono_map"):
        pd.testing.assert_frame_equal(first_state["type_deviation"][key],
                                      second_state["type_deviation"][key])


def test_clone_keeps_all_options():
    from sklearn.base import clone
    from pokeml.models.estimators import Cat_Trainer, LGBM_Trainer
    for model in [Cat_Trainer(cat_features=["stage"], random_seed=42, depth=4),
                  LGBM_Trainer(max_depth=4, min_child_samples=90, random_state=42)]:
        assert clone(model).get_params() == model.get_params()


def test_generation_split_excludes_future_and_target_components():
    from pokeml.pipeline.prepare import prepare_data_train
    data, _ = prepare_data_train(Path(__file__).resolve().parents[1] / "datasets/pkdx_clean.csv",
                                 validation_generation=9)
    assert data.raw_train.generation.max() == 8
    assert set(data.raw_validation.generation) == {9}
    for name in data:
        assert not {"total_stats", "hp", "generation", "base_experience"} & set(data[name][0])


def test_model_round_trip(tmp_path):
    import joblib
    import numpy as np
    from pokeml.models.factory import make_estimator
    data, _ = prepare_data_train(Path(__file__).resolve().parents[1] / "datasets/pkdx_min.csv")
    for name, (X_train, X_valid, y_train, _, cats) in data.items():
        params = ({"iterations": 120, "depth": 3, "use_best_model": False}
                  if name.startswith("cat_") else ({"n_estimators": 20, "max_depth": 3} if name == "xgboost"
                        else {"n_estimators": 20, "num_leaves": 7}))
        model = make_estimator(name, params, cats).fit(X_train, y_train)
        path = tmp_path / f"{name}.joblib"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        np.testing.assert_allclose(model.predict(X_valid), loaded.predict(X_valid))
        assert len(loaded.predict_distribution(X_valid)["pred_bst"]) == len(X_valid)
