"""Forward-cohort comparisons fit complete estimators on earlier generations only."""

import numpy as np
from pokeml.evaluation.tuning import run_search

from pokeml.evaluation.metrics import regression_metrics
from pokeml.models.baseline import StageMedianBaseline
from pokeml.models.deployment import PokemonPredictor
from pokeml.models.factory import make_estimator
from pokeml.models.residual import BaselineCorrectionRegressor


VALIDATION_GENERATIONS = (6, 7, 8, 9)


def forward_splits(rows, generations=VALIDATION_GENERATIONS):
    """Explicit expanding windows; no random generation mixing or future labels."""
    if "generation" not in rows or rows.generation.isna().any():
        raise ValueError("Forward evaluation requires generation metadata")
    splits = []
    for generation in generations:
        train = np.flatnonzero((rows.generation < generation).to_numpy())
        valid = np.flatnonzero((rows.generation == generation).to_numpy())
        if len(train) == 0 or len(valid) == 0:
            raise ValueError(f"Missing training/validation rows for generation {generation}")
        splits.append((train, valid))
    return splits



def pooled_summary(metrics):
    """Each validation Pokémon gets one vote; pool squared errors before RMSE.

    Per-generation metrics stay intact. Counts refer only to validation rows,
    never training population or complete evolution-family sizes.
    """
    work = metrics.copy()
    if (work.n_validation <= 0).any():
        raise ValueError("Validation counts must be positive")
    work["absolute_error_sum"] = work.MAE * work.n_validation
    work["squared_error_sum"] = work.RMSE ** 2 * work.n_validation
    keys = ["population", "model"] + (["evaluation_scope"] if "evaluation_scope" in work else [])
    result = work.groupby(keys, observed=True).agg(
        absolute_error_sum=("absolute_error_sum", "sum"),
        squared_error_sum=("squared_error_sum", "sum"),
        n_validation=("n_validation", "sum"),
        generations=("generation", "nunique"), folds=("generation", "size")).reset_index()
    if (result.generations != result.folds).any():
        raise ValueError("Duplicate generation metrics for model/population")
    result["MAE"] = result.absolute_error_sum / result.n_validation
    result["RMSE"] = np.sqrt(result.squared_error_sum / result.n_validation)
    return result[keys + ["MAE", "RMSE", "n_validation", "generations", "folds"]]



def candidates(parameters, steps, random_state=1, min_count=5):
    """Same attributes and model options for direct and correction comparisons."""
    models = {"stage_median": StageMedianBaseline(min_count)}
    for mode in ("direct", "correction"):
        for variant, options in parameters.items():
            models[f"{mode}_{variant}"] = BaselineCorrectionRegressor(
                variant=variant, estimator=make_estimator(variant, options), steps=steps,
                min_count=min_count, random_state=random_state, mode=mode)
    return models


def tune_candidates(rows, parameters, steps, grid_path, iterations, random_state=1, min_count=5,
                    *, output_directory=None, selection_scope="all"):
    """Search the whole composite inside four forward folds; optimise pooled MAE.

    These are development scores, not an independent assessment after tuning.
    Generations 6–9 all participate; no separate final holdout remains.
    """
    return run_search(rows, parameters, steps, grid_path, iterations, random_state,
                      min_count, output_directory=output_directory, selection_scope=selection_scope)


def evaluate_cohort(model, train, validation, model_name, generation, population="selected_forms"):
    """Pure table outputs; no hard-coded artifact directory or plotting side effects."""
    model.fit(train, train.total_stats)
    predictor = PokemonPredictor(model, model_name)
    predictions = predictor.predict(validation)
    predictions["model"] = model_name
    predictions["actual_bst"] = validation.total_stats.to_numpy()
    predictions["error_bst"] = predictions.actual_bst - predictions.pred_bst
    predictions["absolute_error"] = predictions.error_bst.abs()
    predictions["actual_correction_bst"] = (predictions.actual_bst - predictions.baseline_bst)
    predictions["generation"] = generation
    predictions["rarity"] = validation.rarity.to_numpy()
    predictions["population"] = population
    from pokeml.evaluation.groups import design_groups
    predictions["species_id"] = validation.species_id.to_numpy()
    predictions["design_group"] = design_groups(validation).to_numpy()
    predictions["known_family"] = validation.evolution_chain_id.isin(
        train.evolution_chain_id).to_numpy()
    metric = {"model": model_name, "generation": generation, "population": population,
              "n_train": len(train), "n_validation": len(validation),
              "known_family_rows": int(predictions.known_family.sum()),
              **regression_metrics(validation.total_stats, predictions.pred_bst)}
    if "lower_80" in predictions:
        metric["coverage_80"] = float(((predictions.actual_bst >= predictions.lower_80) &
                                        (predictions.actual_bst <= predictions.upper_80)).mean())
        metric["quantile_crossing_rate"] = float(predictions.quantile_crossed.mean())
    return metric, predictions


def stage_errors(predictions):
    """Report counts alongside errors; small buckets never masquerade as certainty."""
    return predictions.groupby(["population", "generation", "model", "rarity", "stage"],
                               observed=True).agg(
        count=("absolute_error", "size"), MAE=("absolute_error", "mean"),
        bias=("error_bst", "mean")).reset_index()
