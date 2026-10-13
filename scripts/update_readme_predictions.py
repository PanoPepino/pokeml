"""Refit top three saved configurations and update README predictions; no new tuning."""
import argparse
from hashlib import sha256
import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from pokeml.evaluation.groups import grouped_metrics, pooled_group_summary
from pokeml.features.feature_registry import get_feature_steps
from pokeml.models.deployment import PokemonPredictor
from pokeml.models.factory import make_estimator
from pokeml.models.residual import BaselineCorrectionRegressor
from pokeml.pipeline.workflow import load_historical_workflow_result, package_source_digest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('browt', 'pombon', 'gecqua')
LABELS = {'direct_cat_native': 'CatBoost native — direct',
          'direct_cat_ordinal': 'CatBoost ordinal — direct',
          'direct_light_gbm': 'LightGBM — direct', 'direct_xgboost': 'XGBoost — direct',
          'correction_cat_native': 'CatBoost native — correction',
          'correction_cat_ordinal': 'CatBoost ordinal — correction',
          'correction_light_gbm': 'LightGBM — correction',
          'correction_xgboost': 'XGBoost — correction'}
START, END = '<!-- PREDICTIONS -->', '<!-- /PREDICTIONS -->'


def comparison(parent_directory, output_directory):
    parent, output = Path(parent_directory).resolve(), Path(output_directory).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Comparison must be outside original run')
    load_historical_workflow_result(parent)  # Check completed run and both dataset hashes.
    parent_manifest_path = next(parent.glob('*_model_manifest.json'))
    metadata = json.loads(parent_manifest_path.read_text())
    details_path = parent / 'development_details.csv'
    summary = pooled_group_summary(grouped_metrics(pd.read_csv(details_path)))
    ranking = summary.query("population=='selected_forms' and evaluation_scope=='ordinary' and model!='stage_median'")
    ranking = ranking.sort_values(['MAE', 'RMSE', 'model'], kind='stable').head(3)
    if len(ranking) != 3:
        raise ValueError('Three saved regression candidates are required')
    training = pd.read_csv(metadata['training_csv'])
    training = training.loc[training.generation <= metadata['validation_generation']]
    announcements = pd.read_csv(metadata['prediction_csv'])
    if not all(name in announcements.name.tolist() for name in NAMES):
        raise ValueError('Announcement data must include all three README Pokémon')
    inputs = dict(parent_manifest_sha256=sha256(parent_manifest_path.read_bytes()).hexdigest(),
                  predictions_sha256=sha256(details_path.read_bytes()).hexdigest(),
                  training_sha256=metadata['training_sha256'],
                  announcement_sha256=metadata['prediction_sha256'],
                  source_sha256=package_source_digest(),
                  fit_code_sha256=sha256(inspect.getsource(comparison).encode()).hexdigest(),
                  models=ranking.model.tolist(),
                  parameters={name: metadata['parameters'][name] for name in ranking.model})
    manifest_path = output / 'manifest.json'
    if output.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'complete' or manifest.get('inputs') != inputs:
            raise ValueError('Existing comparison does not match; choose a new output directory')
        for filename, digest in manifest['files'].items():
            if sha256((output / filename).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Saved comparison file changed: {filename}')
        return pd.read_csv(output / 'predictions.csv'), pd.read_csv(output / 'ranking.csv')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', inputs=inputs,
                    trained_through_generation=metadata['validation_generation'],
                    n_train=len(training),
                    note='Ranked by ordinary errors from saved experiments; refitted frozen settings; no new tuning or validation.')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        steps = get_feature_steps('custom', metadata['feature_steps'])
        predictions = []
        for name in ranking.model:
            print(f'Refitting {name} on {len(training)} Pokémon', flush=True)
            mode, variant = name.split('_', 1)
            model = BaselineCorrectionRegressor(variant, make_estimator(variant, metadata['parameters'][name]),
                                               steps, metadata['baseline_min_count'],
                                               metadata.get('random_state', 1), mode).fit(training, training.total_stats)
            directory = output / 'models' / name
            PokemonPredictor(model, f'{output.name}/{name}',
                             dict(trained_through_generation=metadata['validation_generation'],
                                  parameters=metadata['parameters'][name], n_train=len(training))).save(directory)
            result = PokemonPredictor.load(directory).predict(announcements)
            np.testing.assert_allclose(result.pred_bst, model.predict(announcements))
            result['model'] = name
            predictions.append(result)
        table = pd.concat(predictions, ignore_index=True)
        table.to_csv(output / 'predictions.csv', index=False)
        ranking.to_csv(output / 'ranking.csv', index=False)
        manifest['files'] = {str(p.relative_to(output)): sha256(p.read_bytes()).hexdigest()
                             for p in output.rglob('*') if p.is_file() and p != manifest_path}
        manifest['status'] = 'complete'
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return table, ranking
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise


def prediction_cell(row):
    if pd.notna(row.get('lower_80')):
        outside = not row.lower_80 <= row.pred_bst <= row.upper_80
        marker = ' †' if outside else ''
        return f'{row.pred_bst:.0f} [{row.lower_80:.0f}, {row.upper_80:.0f}]{marker}'
    return f'{row.pred_bst:.0f} ± {row.predictive_std:.0f}'


def prediction_table(predictions, ranking):
    lines = ['| Model | Ordinary validation MAE | Browt | Pombon | Gecqua |',
             '|---|---:|---:|---:|---:|']
    for row in ranking.itertuples():
        values = predictions.loc[predictions.model.eq(row.model)].set_index('name')
        if not values.index.is_unique:
            raise ValueError('Duplicate prediction names')
        cells = [prediction_cell(values.loc[name]) for name in NAMES]
        lines.append(f'| {LABELS[row.model]} | {row.MAE:.2f} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def update_readme(path, table):
    path = Path(path)
    text = path.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('README must contain one prediction marker pair')
    start, end = text.index(START) + len(START), text.index(END)
    if end < start:
        raise ValueError('README prediction markers are out of order')
    path.write_text(text[:start] + '\n\n' + table + '\n\n' + text[end:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, default=ROOT / 'artifacts/workflow/pokemon_tuned_60_v1')
    parser.add_argument('--output-directory', type=Path, default=ROOT / 'artifacts/predictions/readme_top_three_v2')
    parser.add_argument('--readme', type=Path, default=ROOT / 'README.md')
    args = parser.parse_args()
    predictions, ranking = comparison(args.run_directory, args.output_directory)
    table = prediction_table(predictions, ranking)
    update_readme(args.readme, table)
    print(table)


if __name__ == '__main__':
    main()
