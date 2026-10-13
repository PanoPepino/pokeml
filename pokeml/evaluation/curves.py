"""Training diagnostics from fixed tree prefixes; never select or truncate models."""
from hashlib import sha256
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pokeml.evaluation.metrics import regression_metrics


def checkpoints(total, maximum=200):
    if total < 1 or maximum < 2:
        raise ValueError('Require fitted trees and at least two checkpoints')
    return np.unique(np.linspace(1, total, min(total, maximum), dtype=int)).tolist()


def collect_curves(model, train, validation, name, generation, maximum=200):
    """Score reconstructed BST, using inference-time training predictions too.

    Training errors are optimistic by definition. Validation rows never fit any
    state. Fixed-prefix diagnostics do not trigger early stopping or selection.
    """
    iterations = checkpoints(model.get_iteration_count(), maximum)
    records = []
    for split, frame in (('train', train), ('validation', validation)):
        target = frame.total_stats.to_numpy()
        reference = regression_metrics(target, model.baseline_.predict(frame))
        final = None
        for iteration, prediction in model.predict_at_iterations(frame, iterations):
            scores = regression_metrics(target, prediction)
            records.append(dict(model=name, generation=generation, split=split,
                                iteration=iteration, n_rows=len(frame), MAE=scores['MAE'],
                                RMSE=scores['RMSE'], baseline_MAE=reference['MAE'],
                                baseline_RMSE=reference['RMSE']))
            final = prediction
        if not np.allclose(final, model.predict(frame), rtol=1e-8, atol=1e-8):
            raise ValueError(f'{name} generation {generation}: final prefix differs from prediction')
    return pd.DataFrame(records)


def curve_summary(curves):
    rows = []
    for (name, generation), group in curves.groupby(['model', 'generation'], sort=False):
        train = group.loc[group.split == 'train'].sort_values('iteration').set_index('iteration')
        valid = group.loc[group.split == 'validation'].sort_values('iteration').set_index('iteration')
        end = valid.iloc[-1]
        final_train = train.iloc[-1]
        row = dict(model=name, generation=generation, fitted_rounds=int(valid.index[-1]),
                   checkpoints=len(valid), train_rows=int(final_train.n_rows),
                   validation_rows=int(end.n_rows))
        for metric in ('MAE', 'RMSE'):
            best_round = int(valid[metric].idxmin())
            row.update({f'final_train_{metric}': final_train[metric],
                        f'final_validation_{metric}': end[metric],
                        f'{metric}_gap': end[metric] - final_train[metric],
                        f'min_validation_{metric}': valid.loc[best_round, metric],
                        f'min_{metric}_round': best_round,
                        f'{metric}_deterioration': end[metric] - valid.loc[best_round, metric],
                        f'train_{metric}_change_after_min': final_train[metric] - train.loc[best_round, metric]})
        rows.append(row)
    return pd.DataFrame(rows)


def save_curve_outputs(curves, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    curves.to_csv(directory / 'curves.csv', index=False)
    summary = curve_summary(curves)
    summary.to_csv(directory / 'summary.csv', index=False)
    plot_dir = directory / 'plots'
    plot_dir.mkdir(exist_ok=True)
    for generation, cohort in curves.groupby('generation', sort=True):
        names = list(cohort.model.unique())
        for metric in ('MAE', 'RMSE'):
            ncols = min(4, len(names))
            nrows = (len(names) + ncols - 1) // ncols
            fig = Figure(figsize=(5.2 * ncols, 4.2 * nrows))
            FigureCanvasAgg(fig)
            axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
            for ax, name in zip(axes, names):
                for split, colour in (('train', '#2878B5'), ('validation', '#D55E00')):
                    points = cohort.loc[(cohort.model == name) & (cohort.split == split)].sort_values('iteration')
                    ax.plot(points.iteration, points[metric], color=colour, label=split)
                    ax.axhline(points[f'baseline_{metric}'].iloc[0], color=colour,
                               linestyle='--', alpha=0.45, label=f'{split} baseline')
                ax.set(title=name, xlabel='Boosting rounds', ylabel=f'BST {metric}')
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
            for ax in axes[len(names):]:
                ax.set_visible(False)
            fig.suptitle(f'Generation {generation}: fixed tuned configurations — {metric}\n'
                         'Sampled tree prefixes; diagnostic curves, not independent holdout', fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(plot_dir / f'generation_{generation}_{metric.lower()}.png', dpi=150)
            fig.clear()
    return summary


def diagnose_run(parent_directory, output_directory, maximum=200):
    """Read historical run without weakening normal workflow cache checks.

    Earlier folds refit frozen parameters under current code; generation 9 reuses
    saved diagnostic models. Endpoint reproduction gates completion. Parent run
    and release remain immutable; output must use a new separate directory.
    """
    from pokeml.models.deployment import PokemonPredictor
    from pokeml.models.factory import make_estimator
    from pokeml.models.residual import BaselineCorrectionRegressor
    from pokeml.features.feature_registry import get_feature_steps
    from pokeml.pipeline.workflow import package_source_digest
    parent = Path(parent_directory).resolve()
    output = Path(output_directory).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Diagnostics must live outside immutable parent run')
    if output.exists():
        raise FileExistsError('Choose a new diagnostics output directory')
    manifests = list(parent.glob('*_model_manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Parent has no unique workflow manifest')
    metadata = json.loads(manifests[0].read_text())
    if metadata.get('status') != 'complete':
        raise ValueError('Parent workflow is incomplete')
    source = Path(metadata['training_csv'])
    if sha256(source.read_bytes()).hexdigest() != metadata['training_sha256']:
        raise ValueError('Parent training dataset changed')
    frame = pd.read_csv(source)
    frame = frame.loc[frame.generation <= metadata['validation_generation']]
    expected = pd.read_csv(parent / 'development_metrics.csv').query("population == 'selected_forms'")
    steps = get_feature_steps('custom', metadata['feature_steps'])
    output.mkdir(parents=True)
    manifest = dict(status='running', parent_directory=str(parent),
                    packages={name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost')},
                    parent_manifest_sha256=sha256(manifests[0].read_bytes()).hexdigest(),
                    parent_source_sha256=metadata['source_sha256'],
                    source_sha256=package_source_digest(), training_sha256=metadata['training_sha256'],
                    parameters=metadata['parameters'], validation_generations=metadata['validation_generations'],
                    maximum_checkpoints=maximum, population='selected_forms', early_stopping=False,
                    note='Frozen tuned parameters; sampled prefixes; no independent holdout.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    tables = []
    try:
        for generation in metadata['validation_generations']:
            train, valid = frame.loc[frame.generation < generation], frame.loc[frame.generation == generation]
            for name, params in metadata['parameters'].items():
                mode, variant = name.split('_', 1)
                print(f'Curves {name}, validation generation {generation}', flush=True)
                if generation == metadata['validation_generation']:
                    model = PokemonPredictor.load(parent / 'evaluation_models' / name).model
                else:
                    params = dict(params)
                    if variant in ('light_gbm', 'xgboost'):
                        params['fit_uncertainty'] = False
                    model = BaselineCorrectionRegressor(variant, make_estimator(variant, params), steps,
                                                       metadata['baseline_min_count'],
                                                       metadata.get('random_state', 1), mode)
                    model.fit(train, train.total_stats)
                table = collect_curves(model, train, valid, name, generation, maximum)
                endpoint = table.query("split == 'validation'").iloc[-1]
                original = expected.loc[(expected.model == name) & (expected.generation == generation)]
                if len(original) != 1 or not np.allclose(endpoint[['MAE', 'RMSE']].to_numpy(dtype=float),
                                                       original[['MAE', 'RMSE']].iloc[0].to_numpy(dtype=float),
                                                       rtol=1e-7, atol=1e-7):
                    raise ValueError(f'{name} generation {generation}: saved scores do not reproduce')
                tables.append(table)
            pd.concat(tables, ignore_index=True).to_csv(output / 'curves.csv', index=False)
        summary = save_curve_outputs(pd.concat(tables, ignore_index=True), output)
        manifest['status'] = 'complete'
        manifest['endpoint_scores_verified'] = True
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        return summary
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        raise
