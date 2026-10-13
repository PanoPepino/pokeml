"""Behaviour checks for fold-local targets and deployable BST reconstruction."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV

from pokeml.models.baseline import StageMedianBaseline
from pokeml.models.deployment import PokemonPredictor
from pokeml.models.factory import make_estimator
from pokeml.models.residual import BaselineCorrectionRegressor
from pokeml.features.feature_registry import get_feature_steps
from pokeml.evaluation.forward import forward_splits


@pytest.fixture
def rows():
    frame = pd.read_csv("datasets/pkdx_min.csv").head(80).copy()
    frame.index = np.arange(80)[::-1]  # Positional target alignment, not accidental label alignment.
    return frame


def test_median_support_and_fallback():
    train = pd.DataFrame({"rarity": ["regular"] * 6, "stage": ["single"] * 5 + ["s1c2"]})
    model = StageMedianBaseline().fit(train, [400, 410, 420, 430, 670, 300])
    query = pd.DataFrame({"rarity": ["regular", "regular", "legendary"],
                          "stage": ["single", "s1c2", "s3c3"]}, index=[4, 4, 2])
    values = model.predict_components(query)
    np.testing.assert_allclose(values.pred_bst, [420, 415, 415])
    assert list(values.baseline_source) == ["stage_rarity", "rarity", "global"]
    assert list(values.baseline_count) == [5, 1, 0]
    assert list(values.index) == [4, 4, 2]
    np.testing.assert_allclose(values.correction_bst, 0)
    with pytest.raises(ValueError, match="stage"):
        model.predict(query.assign(stage="unknown"))


@pytest.mark.parametrize("variant", ["cat_native", "cat_ordinal", "light_gbm", "xgboost"])
def test_reconstruction_distribution_and_release(rows, variant, tmp_path):
    params = ({"iterations": 120, "depth": 3} if variant.startswith("cat_")
              else ({"n_estimators": 15, "max_depth": 3} if variant == "xgboost"
                    else {"n_estimators": 15, "num_leaves": 7}))
    model = BaselineCorrectionRegressor(variant, make_estimator(variant, params),
                                       get_feature_steps("custom", ["initial_tag"]))
    cloned = clone(model)
    assert cloned.estimator.get_params() == model.estimator.get_params()
    assert cloned.variant == model.variant and cloned.steps == model.steps
    train, query = rows.iloc[:60], rows.iloc[60:].drop(columns="total_stats")
    model.fit(train, train.total_stats)
    baseline = model.baseline_.predict(train)
    np.testing.assert_allclose(model.training_residuals_, train.total_stats.to_numpy() - baseline)
    components = model.predict_components(query)
    np.testing.assert_allclose(components.pred_bst, components.baseline_bst + components.correction_bst)
    raw = model.model_.predict_distribution(model.preprocessor_.transform(query))
    shifted = model.predict_distribution(query)
    np.testing.assert_allclose(shifted["pred_bst"], raw["pred_bst"] + components.baseline_bst)
    if "lower_80" in raw:
        np.testing.assert_allclose(shifted["lower_80"], raw["lower_80"] + components.baseline_bst)
        np.testing.assert_allclose(shifted["upper_80"], raw["upper_80"] + components.baseline_bst)
    else:
        np.testing.assert_allclose(shifted["predictive_std"], raw["predictive_std"])
    predictor = PokemonPredictor(model, "test/v1")
    output = predictor.predict(query)
    assert list(output.name) == list(query.name)
    predictor.save(tmp_path / "release")
    pd.testing.assert_frame_equal(output, PokemonPredictor.load(tmp_path / "release").predict(query))
    pd.testing.assert_frame_equal(output, predictor.predict(query.assign(total_stats=99999)))
    with pytest.raises(ValueError, match="height"):
        predictor.predict(query.assign(height=-1))


def test_holdout_targets_do_not_affect_training(rows):
    train, valid = rows.iloc[:60], rows.iloc[60:].copy()
    def fit():
        return BaselineCorrectionRegressor(estimator=make_estimator("cat_native", {"iterations": 120, "depth": 3})).fit(train, train.total_stats)
    first = fit()
    before = first.predict(valid)
    valid["total_stats"] += 5000
    second = fit()
    pd.testing.assert_frame_equal(first.baseline_.buckets_, second.baseline_.buckets_)
    np.testing.assert_allclose(first.training_residuals_, second.training_residuals_)
    np.testing.assert_allclose(before, second.predict(valid))


def test_forward_search_owns_baseline(rows):
    rows["generation"] = np.repeat([5, 6, 7, 8, 9], [20, 5, 10, 15, 30])
    rows["total_stats"] = rows.total_stats + (rows.generation - 5) * 100
    splits = forward_splits(rows)
    for tr, va in splits:
        assert rows.iloc[tr].generation.max() < rows.iloc[va].generation.min()
        local = StageMedianBaseline().fit(rows.iloc[tr], rows.iloc[tr].total_stats)
        assert local.global_median_ == rows.iloc[tr].total_stats.median()
        assert local.global_median_ != rows.total_stats.median()
    model = BaselineCorrectionRegressor(estimator=make_estimator("light_gbm", {"n_estimators": 10}), variant="light_gbm")
    search = RandomizedSearchCV(model, {"estimator__num_leaves": [3, 7]}, n_iter=1,
                                cv=splits, scoring="neg_mean_absolute_error", error_score="raise")
    search.fit(rows, rows.total_stats)
    assert search.best_estimator_.baseline_.global_median_ == rows.total_stats.median()
    assert np.isfinite(search.best_score_)


def test_release_rejects_legacy_format(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format_version": 0}))
    with pytest.raises(ValueError, match="Incompatible"):
        PokemonPredictor.load(tmp_path)


def test_direct_mode_reports_reference_difference(rows):
    model = BaselineCorrectionRegressor(variant="light_gbm", mode="direct",
                                       estimator=make_estimator("light_gbm", {"n_estimators": 10})).fit(
                                           rows, rows.total_stats)
    parts = model.predict_components(rows)
    np.testing.assert_allclose(parts.baseline_bst, model.baseline_.predict(rows))
    np.testing.assert_allclose(parts.pred_bst, model.model_.predict(model.preprocessor_.transform(rows)))
    np.testing.assert_allclose(parts.pred_bst, parts.baseline_bst + parts.correction_bst)
    assert set(parts.prediction_mode) == {"direct"}


def test_workflow_release_and_csv_command(tmp_path):
    from hashlib import sha256
    from typer.testing import CliRunner
    from pokeml.cli import app
    from pokeml.pipeline.workflow import WorkflowConfig, run_workflow, load_workflow_result
    source = pd.read_csv("datasets/pkdx_min.csv")
    # Small real cohort retains every forward window and starter prediction inputs.
    cohort = source.groupby("generation", group_keys=False).head(30)
    training = tmp_path / "training.csv"
    cohort.to_csv(training, index=False)
    hashes = {p: sha256(p.read_bytes()).hexdigest() for p in
              (training, Path("datasets/new_pokes.csv"))}
    result = run_workflow(WorkflowConfig(training_csv=str(training), output_root=str(tmp_path),
                                        run_name="check", sensitivity=False,
                                        parameters={"light_gbm": {"n_estimators": 10, "num_leaves": 3}}))
    replay = load_workflow_result(result.run_directory)
    assert replay.selected_model == result.selected_model
    assert set(result.development_metrics.generation) == {6, 7, 8, 9}
    assert set(result.development_summary.folds) == {4}
    assert set(result.development_summary.n_validation) == {120}
    individual = pd.read_csv(result.run_directory / "development_details.csv")
    assert not individual.duplicated(["population", "generation", "model", "name"]).any()
    assert result.stage_diagnostics["count"].sum() == len(individual)
    last_fold = individual.loc[individual.generation == 9].reset_index(drop=True)
    pd.testing.assert_frame_equal(last_fold, pd.read_csv(result.run_directory / "validation_details.csv"))
    for row in result.development_summary.itertuples():
        errors = individual.loc[individual.model == row.model, "error_bst"]
        assert row.MAE == pytest.approx(errors.abs().mean())
        assert row.RMSE == pytest.approx(np.sqrt((errors ** 2).mean()))
    pd.testing.assert_frame_equal(replay.prediction_details, pd.read_csv(result.run_directory / "predictions_details.csv"))
    manifest = json.loads((result.release_directory / "manifest.json").read_text())
    assert manifest["validation_generations"] == [6, 7, 8, 9]
    assert manifest["selection_weighting"] == "pokemon"
    assert manifest["separate_holdout"] is False
    assert manifest["trained_through_generation"] == 9
    assert manifest["n_train"] == len(cohort)
    for model in ("stage_median", "direct_light_gbm", "correction_light_gbm"):
        m = json.loads((result.run_directory / "evaluation_models" / model / "manifest.json").read_text())
        assert m["trained_through_generation"] == 8
    output = tmp_path / "predictions.csv"
    command = CliRunner().invoke(app, ["model", "predict-release", "--release-directory",
                                     str(result.release_directory), "--output-path", str(output)])
    assert command.exit_code == 0, command.output
    pd.testing.assert_frame_equal(pd.read_csv(output).drop(columns="model", errors="ignore"),
                                  replay.prediction_details.drop(columns="model"))
    command = CliRunner().invoke(app, ["model", "predict-release", "--release-directory",
                                     str(result.release_directory), "--output-path", str(output)])
    assert command.exit_code != 0  # Existing output cannot overwrite input/data artifacts.
    assert all(sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    manifest_path = result.run_directory / f"{result.run_directory.name}_model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = "stale"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source changed"):
        load_workflow_result(result.run_directory)


@pytest.mark.parametrize("variant", ["cat_native", "cat_ordinal", "light_gbm", "xgboost"])
def test_optional_forward_tuning(rows, tmp_path, variant):
    from pokeml.evaluation.forward import tune_candidates
    rows["generation"] = np.repeat([5, 6, 7, 8, 9], [20, 5, 10, 15, 30])
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"common_params": {}, "params_grid": {
        "catboost": {"depth": [3, 4]}, "light_gbm": {"num_leaves": [3, 7]}, "xgboost": {"max_depth": [2, 3]}}}))
    options = {"iterations": 120} if variant.startswith("cat_") else {"n_estimators": 10}
    models, scores = tune_candidates(rows, {variant: options}, [], grid, 1)
    assert set(scores.model) == {f"direct_{variant}", f"correction_{variant}"}
    assert not hasattr(models[f"correction_{variant}"], "baseline_")
    key = "iterations" if variant.startswith("cat_") else "n_estimators"
    assert models[f"correction_{variant}"].estimator.get_params()[key] == options[key]


def test_zero_correction_reproduces_baseline(rows, monkeypatch):
    model = BaselineCorrectionRegressor(variant="light_gbm",
                                       estimator=make_estimator("light_gbm", {"n_estimators": 10})).fit(
                                           rows, rows.total_stats)
    monkeypatch.setattr(model.model_, "predict", lambda encoded: np.zeros(len(encoded)))
    np.testing.assert_allclose(model.predict(rows), model.baseline_.predict(rows))




def test_pooled_summary_matches_individual_predictions():
    from pokeml.evaluation.forward import pooled_summary
    errors = [np.array([1.0]), np.array([2.0, 4.0]), np.array([5.0]),
              np.array([0.0, 0.0, 0.0, 10.0])]
    metrics = pd.DataFrame([{"population": "selected_forms", "model": "test",
                             "generation": generation, "n_validation": len(error),
                             "MAE": np.abs(error).mean(), "RMSE": np.sqrt(np.mean(error ** 2))}
                            for generation, error in zip((6, 7, 8, 9), errors)])
    result = pooled_summary(metrics).iloc[0]
    pooled = np.concatenate(errors)
    assert result.MAE == pytest.approx(np.abs(pooled).mean())
    assert result.RMSE == pytest.approx(np.sqrt(np.mean(pooled ** 2)))
    assert result.folds == result.generations == 4
    assert result.n_validation == len(pooled)
    assert result.MAE != pytest.approx(metrics.MAE.mean())
    assert result.RMSE != pytest.approx(metrics.RMSE.mean())
    with pytest.raises(ValueError, match="Duplicate generation"):
        pooled_summary(pd.concat([metrics, metrics.iloc[:1]]))


