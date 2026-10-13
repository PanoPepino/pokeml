"""Reproducible random search with explicit defaults and complete trial records."""
from hashlib import sha256
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler, RandomizedSearchCV

from pokeml.artifacts import tuning_section
from pokeml.config import load_tuning_grid
from pokeml.models.factory import make_estimator


def _json(value):
    return json.dumps(value, sort_keys=True, default=lambda item: item.item())


def canonical_parameters(variant, options):
    """Normalize historical aliases, rejecting conflicting names in one mapping."""
    result = dict(options)
    aliases = ({'max_depth': 'depth', 'n_estimators': 'iterations'}
               if variant.startswith('cat_') else {'bagging_fraction': 'subsample',
                                                   'bagging_freq': 'subsample_freq'}
               if variant == 'light_gbm' else {})
    for alias, canonical in aliases.items():
        if alias in result:
            if canonical in result:
                raise ValueError(f'{variant}: choose {canonical} or {alias}, not both')
            result[canonical] = result.pop(alias)
    if result.get('early_stopping_rounds') or result.get('use_best_model'):
        raise ValueError('Scored validation folds cannot select boosting iterations')
    result.pop('early_stopping_rounds', None)
    result.pop('fit_uncertainty', None)
    return result


def sample_configurations(variant, defaults, grid, iterations, seed=1):
    """Default first, then unique samples; same schedule serves both model modes."""
    if iterations < 1:
        raise ValueError('Search budget must be positive')
    defaults = canonical_parameters(variant, defaults)
    grid = canonical_parameters(variant, grid)
    configurations = [defaults]
    seen = {_json(defaults)}
    # One extra sampled configuration covers a possible duplicate of defaults.
    count = min(iterations, len(ParameterGrid(grid)))
    for sample in ParameterSampler(grid, n_iter=count, random_state=seed):
        options = {**defaults, **sample}
        key = _json(options)
        if key not in seen:
            configurations.append(options)
            seen.add(key)
        if len(configurations) == iterations:
            break
    return configurations[:iterations]


def trial_table(results, configurations, counts, model_name):
    """Pool absolute and squared errors; preserve chronological trial identities."""
    weights = np.asarray(counts, dtype=float) / sum(counts)
    identity = {_json(config): i for i, config in enumerate(configurations)}
    records = []
    for position, params in enumerate(results['params']):
        config = {key.removeprefix('estimator__'): value for key, value in params.items()}
        trial_id = identity[_json(config)]
        maes = np.array([-results[f'split{i}_test_MAE'][position] for i in range(len(counts))])
        rmses = np.array([-results[f'split{i}_test_RMSE'][position] for i in range(len(counts))])
        if not np.isfinite(np.r_[maes, rmses]).all():
            raise ValueError('Search scores must be finite')
        row = dict(model=model_name, trial_id=trial_id, is_default=trial_id == 0,
                   params=_json(config), pooled_MAE=float(maes @ weights),
                   pooled_RMSE=float(np.sqrt(rmses ** 2 @ weights)),
                   n_validation=sum(counts), folds=len(counts),
                   mean_fit_seconds=results['mean_fit_time'][position],
                   mean_score_seconds=results['mean_score_time'][position],
                   fit_seconds=results['mean_fit_time'][position] * len(counts))
        for i, generation in enumerate((6, 7, 8, 9)):
            row[f'gen_{generation}_MAE'] = maes[i]
            row[f'gen_{generation}_RMSE'] = rmses[i]
            row[f'gen_{generation}_n_validation'] = counts[i]
        records.append(row)
    table = pd.DataFrame(records).sort_values(['pooled_MAE', 'pooled_RMSE', 'trial_id'], kind='stable')
    table['rank'] = np.arange(1, len(table) + 1)
    return table.sort_values('trial_id').reset_index(drop=True)


def run_search(rows, parameters, steps, grid_path, iterations, seed=1, min_count=5,
               *, output_directory=None, selection_scope="all"):
    """Search point estimates only; caller refits winners with uncertainty enabled.

    Every fold fits its own baseline and transforms. No all-history search refit
    occurs. All four validation generations participate in selection, not holdout.
    """
    from pokeml.evaluation.forward import candidates, forward_splits
    from pokeml.pipeline.workflow import package_source_digest
    grid = load_tuning_grid(grid_path)
    from pokeml.evaluation.groups import selection_mask
    eligible = selection_mask(rows, selection_scope).to_numpy()
    splits = [(train, valid[eligible[valid]]) for train, valid in forward_splits(rows)]
    if any(len(valid) == 0 for _, valid in splits):
        raise ValueError('Selected validation group is empty in a required fold')
    counts = [len(valid) for _, valid in splits]
    schedules = {variant: sample_configurations(variant, options, grid[tuning_section(variant)],
                                               iterations, seed)
                 for variant, options in parameters.items()}
    models = candidates(parameters, steps, seed, min_count)
    summaries, tables, winners = [], [], {}
    directory = Path(output_directory) if output_directory is not None else None
    manifest = {'status': 'running', 'format_version': 1,
                'source_sha256': package_source_digest(),
                'dataset_frame_sha256': sha256(pd.util.hash_pandas_object(rows, index=True).values.tobytes()).hexdigest(),
                'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost')}, 'seed': seed, 'requested_trials': iterations,
                'validation_generations': [6, 7, 8, 9], 'validation_counts': counts,
                'selection': 'pooled Pokemon MAE', 'selection_scope': selection_scope, 'separate_holdout': False,
                'grid_sha256': sha256(Path(grid_path).read_bytes()).hexdigest(),
                'expanded_grid': grid, 'configurations': schedules, 'completed_models': []}
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'manifest.json').write_text(_json(manifest))
    for name, model in models.items():
        if name == 'stage_median':
            continue
        configurations = schedules[model.variant]
        search_only = model.variant in ('light_gbm', 'xgboost')
        if search_only:
            model.estimator.set_params(fit_uncertainty=False)
        distributions = [{f'estimator__{key}': [value] for key, value in config.items()}
                         for config in configurations]
        print(f'Tuning {name}: {len(configurations)} trials × {len(splits)} folds', flush=True)
        start = perf_counter()
        search = RandomizedSearchCV(model, distributions, n_iter=len(configurations),
                                    scoring={'MAE': 'neg_mean_absolute_error',
                                             'RMSE': 'neg_root_mean_squared_error'},
                                    cv=splits, refit=False, random_state=seed, n_jobs=1,
                                    error_score='raise')
        search.fit(rows, rows.total_stats)
        table = trial_table(search.cv_results_, configurations, counts, name)
        table['selection_scope'] = selection_scope
        best = table.loc[table['rank'] == 1].iloc[0]
        options = json.loads(best.params)
        if search_only:
            options['fit_uncertainty'] = True
        # Search estimator has no fitted state; only selected parameters survive.
        model.estimator = make_estimator(model.variant, options)
        winners[name] = options
        summary = dict(model=name, selection_scope=selection_scope, development_MAE=best.pooled_MAE,
                       development_RMSE=best.pooled_RMSE, folds=len(splits),
                       n_validation=sum(counts), trials=len(configurations),
                       best_trial_id=int(best.trial_id), default_won=bool(best.is_default),
                       default_MAE=float(table.loc[table.is_default, 'pooled_MAE'].iloc[0]),
                       elapsed_seconds=perf_counter() - start, params=_json(options))
        summaries.append(summary)
        tables.append(table)
        manifest['completed_models'].append(name)
        if directory:
            table.to_csv(directory / f'{name}_trials.csv', index=False)
            pd.concat(tables, ignore_index=True).to_csv(directory / 'trials.csv', index=False)
            pd.DataFrame(summaries).to_csv(directory / 'summary.csv', index=False)
            (directory / 'best_parameters.json').write_text(_json(winners))
            (directory / 'manifest.json').write_text(_json(manifest))
        print(f'Finished {name}: MAE {best.pooled_MAE:.3f}; '
              f'default {summary["default_MAE"]:.3f}; {summary["elapsed_seconds"]:.1f}s', flush=True)
    manifest['status'] = 'complete'
    if directory:
        (directory / 'manifest.json').write_text(_json(manifest))
    return models, pd.DataFrame(summaries)
