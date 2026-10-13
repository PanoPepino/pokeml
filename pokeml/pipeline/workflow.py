"""Offline baseline + correction workflow; CSV inputs remain unchanged.

This workflow compares complete estimators, evaluates reconstructed BST, then creates a separate
all-history deployment bundle. Current-chain metadata makes evaluation retrospective.
"""

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import platform

import pandas as pd
from sklearn.base import clone

from pokeml.data.load import load_data
from pokeml.evaluation.forward import (candidates, evaluate_cohort, forward_splits,
                                      stage_errors, tune_candidates, pooled_summary, VALIDATION_GENERATIONS)
from pokeml.features.feature_registry import get_feature_steps
from pokeml.models.deployment import PokemonPredictor


def package_source_digest():
    """Hash package paths and bytes so cached runs cannot silently use stale code."""
    root = Path(__file__).resolve().parents[1]
    return sha256(b"".join(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes()
                           for p in sorted(root.rglob("*.py")))).hexdigest()


def default_parameters():
    """Bounded starting configuration, not a claim of optimal hyperparameters."""
    return {
        "cat_native": {"iterations": 400, "depth": 4, "use_best_model": False},
        "cat_ordinal": {"iterations": 400, "depth": 4, "use_best_model": False},
        "light_gbm": {"n_estimators": 400, "num_leaves": 15},
        "xgboost": {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03},
    }


@dataclass
class WorkflowConfig:
    """Explicit run inputs; the notebook is a caller, not a second pipeline."""
    training_csv: str = "datasets/pkdx_min.csv"
    prediction_csv: str = "datasets/new_pokes.csv"
    output_root: str = "artifacts/workflow"
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    feature_mode: str = "custom"
    feature_steps: tuple[str, ...] = ("initial_tag",)
    validation_generation: int = 9
    random_state: int = 1
    tune_iterations: int = 0
    tuning_config: str = "configs/tuning_forward.json"
    parameters: dict = field(default_factory=default_parameters)
    min_baseline_count: int = 5
    sensitivity: bool = True
    training_curves: bool = True
    selection_scope: str = "ordinary"


@dataclass
class WorkflowResult:
    """Reviewable four-fold development, last-fold, sensitivity and deployment outputs."""
    run_directory: Path
    model_prefix: Path
    metrics: pd.DataFrame
    baselines: pd.DataFrame
    predictions: pd.DataFrame
    prediction_details: pd.DataFrame
    tuning_results: pd.DataFrame | None
    development_metrics: pd.DataFrame
    development_summary: pd.DataFrame
    validation_details: pd.DataFrame
    stage_diagnostics: pd.DataFrame
    baseline_table: pd.DataFrame
    selected_model: str
    release_directory: Path
    evaluation_metrics: pd.DataFrame | None = None
    evaluation_summary: pd.DataFrame | None = None
    evaluation_stage_errors: pd.DataFrame | None = None
    selection_scope: str = "all"


def run_workflow(config=None):
    """Compare four folds → choose on pooled Pokémon MAE → refit → predict.

    Generations 6–9 all select model and optional hyperparameters. Generation 9
    remains a separately displayed last-fold diagnostic, not a final holdout.
    No validation cohort selects boosting iteration count by early stopping.
    Sensitivity scores are separately refitted on default species only.
    """
    config = config or WorkflowConfig()
    from pokeml.evaluation.groups import selection_mask, grouped_metrics, pooled_group_summary
    selection_mask(pd.DataFrame({"species_id": []}), config.selection_scope)
    if config.validation_generation != 9:
        raise ValueError("Workflow uses four validation generations 6–9; cutoff must be generation 9")
    if Path(config.run_name).name != config.run_name or config.run_name in ("", ".", ".."):
        raise ValueError("run_name must be a single directory name")
    directory = Path(config.output_root) / config.run_name
    if directory.exists():
        raise FileExistsError(f"Choose a new run_name; run already exists: {directory}")
    frame = load_data(config.training_csv)
    announced = load_data(config.prediction_csv)
    required = {"generation", "id", "species_id", "evolution_chain_id", "total_stats"}
    if not required.issubset(frame):
        raise ValueError(f"Training snapshot missing metadata: {sorted(required - set(frame))}")
    if frame.generation.isna().any() or frame.total_stats.isna().any():
        raise ValueError("Training generation and BST must be present")
    # Do not accidentally train the deployment model on generations beyond cutoff.
    frame = frame.loc[frame.generation <= config.validation_generation].copy()
    steps = get_feature_steps(config.feature_mode, config.feature_steps)
    if any(step[0] == "type_deviation" for step in steps):
        raise ValueError("Supervised type maps are disabled in baseline + correction v1")
    development = frame.copy()
    eligible = selection_mask(development, config.selection_scope).to_numpy()
    if any(not eligible[valid].any() for _, valid in forward_splits(development)):
        raise ValueError('Selected validation group is empty in a required fold')
    params = {key: dict(value) for key, value in config.parameters.items()}
    tuning_table = None
    if config.tune_iterations < 0:
        raise ValueError("tune_iterations must be nonnegative")
    directory.mkdir(parents=True)
    if config.tune_iterations:
        models, tuning_table = tune_candidates(development, params, steps, config.tuning_config,
                                              config.tune_iterations, config.random_state,
                                              config.min_baseline_count,
                                              output_directory=directory / "tuning", selection_scope=config.selection_scope)
    else:
        models = candidates(params, steps, config.random_state, config.min_baseline_count)
    dev_metrics, last_fold_metrics, details, dev_details = [], [], [], []
    populations = {"selected_forms": frame}
    if config.sensitivity:
        populations["default_species"] = frame.loc[frame.id == frame.species_id]
    curve_tables = []
    last_fold_models = {}
    for population, rows in populations.items():
        for generation in VALIDATION_GENERATIONS:
            train = rows.loc[rows.generation < generation]
            valid = rows.loc[rows.generation == generation]
            for name, prototype in models.items():
                model = clone(prototype)
                metric, prediction = evaluate_cohort(model, train, valid, name, generation, population)
                if config.training_curves and population == "selected_forms" and name != "stage_median":
                    from pokeml.evaluation.curves import collect_curves
                    curve_tables.append(collect_curves(model, train, valid, name, generation))
                dev_metrics.append(metric)
                dev_details.append(prediction)
                if generation == config.validation_generation:
                    last_fold_metrics.append(metric)
                    details.append(prediction)
                    if population == "selected_forms":
                        last_fold_models[name] = model
    if curve_tables:
        from pokeml.evaluation.curves import save_curve_outputs
        save_curve_outputs(pd.concat(curve_tables, ignore_index=True), directory / "training_curves")
    dev_table = pd.DataFrame(dev_metrics)
    summary = pooled_summary(dev_table)
    # Each validation Pokémon gets one vote. Tie-break favours simpler baseline.
    developmental = pd.concat(dev_details, ignore_index=True)
    group_metrics = grouped_metrics(developmental)
    group_summary = pooled_group_summary(group_metrics)
    group_stages = grouped_metrics(developmental, by_stage=True)
    ranking = group_summary.loc[(group_summary.population == "selected_forms") &
                                (group_summary.evaluation_scope == config.selection_scope)].sort_values(
        ["MAE", "RMSE", "model"], kind="stable")
    best_error = ranking.MAE.min()
    selected = ("stage_median" if (ranking.loc[ranking.model == "stage_median", "MAE"]
                                    <= best_error + 1e-10).any() else ranking.iloc[0].model)
    metrics = pd.DataFrame(last_fold_metrics)
    validation = pd.concat(details, ignore_index=True)
    developmental = pd.concat(dev_details, ignore_index=True)
    stage_table = stage_errors(developmental)
    reference_model = last_fold_models["stage_median"]
    baseline_table = reference_model.buckets_.reset_index()
    baseline_table["uses_fallback"] = baseline_table["count"] < config.min_baseline_count
    # Persist evaluation models separately from release. No all-history model is scored on its generation-9 training rows.
    metadata = {
        "training_csv": str(config.training_csv),
        "training_sha256": sha256(Path(config.training_csv).read_bytes()).hexdigest(),
        "prediction_csv": str(config.prediction_csv),
        "prediction_sha256": sha256(Path(config.prediction_csv).read_bytes()).hexdigest(),
        "source_sha256": package_source_digest(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ("pandas", "numpy", "scikit-learn", "catboost", "lightgbm", "xgboost", "joblib")},
        "feature_steps": [step[0] for step in steps],
        "selected_model": selected, "selection": "Pooled individual-Pokémon MAE across validation generations 6–9",
        "selection_weighting": "pokemon", "selection_scope": config.selection_scope, "validation_generations": list(VALIDATION_GENERATIONS),
        "separate_holdout": False,
        "selection_validation_counts": {str(row.generation): int(row.n_validation)
            for row in group_metrics.loc[(group_metrics.population == "selected_forms") &
                (group_metrics.model == "stage_median") &
                (group_metrics.evaluation_scope == config.selection_scope)].itertuples()},
        "training_curves": config.training_curves,
        "tuned": bool(config.tune_iterations),
        "tune_iterations": config.tune_iterations, "random_state": config.random_state,
        "tuning_config": config.tuning_config,
        "tuning_config_sha256": sha256(Path(config.tuning_config).read_bytes()).hexdigest() if config.tune_iterations else None,
        "parameters": {name: prototype.estimator.get_params() for name, prototype in models.items()
                       if hasattr(prototype, "estimator")},
        "validation_generation": 9,
        "evaluation_note": "Retrospective current-chain/current-stat snapshot; all four folds participate in selection; no separate holdout.",
        "stage_schema": "Current-snapshot route buckets; stage supplied at prediction time",
        "baseline_min_count": config.min_baseline_count,
        "training_population": "selected_forms",
    }
    for name, model in last_fold_models.items():
        PokemonPredictor(model, f"{config.run_name}/evaluation/{name}",
                         {**metadata, "trained_through_generation": 8, "purpose": "last_fold_diagnostic"}).save(
                             directory / "evaluation_models" / name)
    deployed_model = clone(models[selected]).fit(frame, frame.total_stats)
    deployed_baseline = (deployed_model.baseline_ if hasattr(deployed_model, "baseline_") else deployed_model)
    deployed_baseline.buckets_.reset_index().to_csv(directory / "release_baseline_table.csv", index=False)
    release = directory / "release"
    predictor = PokemonPredictor(deployed_model, f"{config.run_name}/{selected}",
                                 {**metadata, "trained_through_generation": 9, "purpose": "deployment",
                                  "n_train": len(frame)})
    predictor.save(release)
    prediction_details = PokemonPredictor.load(release).predict(announced)
    prediction_details["model"] = selected
    predictions = prediction_details[["name", "baseline_bst", "correction_bst", "pred_bst"]].copy()
    tables = {"metrics": metrics, "development_metrics": dev_table, "development_summary": summary,
              "validation_details": validation, "development_details": developmental,
              "stage_diagnostics": stage_table, "baseline_table": baseline_table,
              "evaluation_metrics": group_metrics, "evaluation_summary": group_summary,
              "evaluation_stage_errors": group_stages,
              "predictions": predictions, "predictions_details": prediction_details}
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    baselines = metrics.loc[metrics.model == "stage_median"].copy()
    baselines.to_csv(directory / "baselines.csv", index=False)
    if tuning_table is not None:
        tuning_table.to_csv(directory / "tuning.csv", index=False)
    metadata["training_indices"] = list(map(str, frame.loc[frame.generation < 9].index))
    metadata["selection_validation_indices"] = list(map(str, frame.loc[frame.generation.isin(VALIDATION_GENERATIONS) & selection_mask(frame, config.selection_scope)].index))
    metadata["validation_indices"] = list(map(str, frame.loc[frame.generation == 9].index))
    prefix = directory / f"{config.run_name}_model"
    from pokeml.visualisation.viz_model.correction import workflow_plots
    workflow_plots(directory, dev_table, validation, selected)
    metadata["status"] = "complete"
    Path(f"{prefix}_manifest.json").write_text(json.dumps(metadata, indent=2))
    return WorkflowResult(directory, prefix, metrics, baselines, predictions, prediction_details,
                          tuning_table, dev_table, summary, validation, stage_table,
                          baseline_table, selected, release, group_metrics, group_summary, group_stages,
                          config.selection_scope)


def load_workflow_result(directory):
    """Replay current-source output tables; reject incomplete or stale runs."""
    return _load_workflow_tables(directory, historical=False)


def load_historical_workflow_result(directory):
    """Read frozen historical tables after dataset checks, without any fitting.

    Source may differ from current code. These are original-run results, not
    evidence of present-code performance. Never use this for training cache reuse.
    """
    return _load_workflow_tables(directory, historical=True)


def _load_workflow_tables(directory, historical):
    directory = Path(directory)
    manifests = list(directory.glob("*_model_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Run has no unique completed manifest")
    metadata = json.loads(manifests[0].read_text())
    if metadata.get("status") != "complete" or (not historical and metadata.get("source_sha256") != package_source_digest()):
        raise ValueError("Run is incomplete or package source changed; execute a new run")
    for kind in ("training", "prediction"):
        if sha256(Path(metadata[f"{kind}_csv"]).read_bytes()).hexdigest() != metadata[f"{kind}_sha256"]:
            raise ValueError(f"Run {kind} dataset changed; execute a new run")
    def read(name):
        return pd.read_csv(directory / f"{name}.csv")
    prefix = Path(str(manifests[0])[:-len("_manifest.json")])
    tuning_path = directory / "tuning.csv"
    from pokeml.evaluation.groups import grouped_metrics, pooled_group_summary
    grouped = grouped_metrics(read("development_details"))
    return WorkflowResult(directory, prefix, read("metrics"), read("baselines"),
                          read("predictions"), read("predictions_details"),
                          pd.read_csv(tuning_path) if tuning_path.exists() else None,
                          read("development_metrics"), read("development_summary"),
                          read("validation_details"), read("stage_diagnostics"), read("baseline_table"),
                          metadata["selected_model"], directory / "release", grouped, pooled_group_summary(grouped),
                          grouped_metrics(read("development_details"), by_stage=True),
                          metadata.get("selection_scope", "all"))
