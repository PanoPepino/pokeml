"""Fixed design groups for scoring; membership never becomes a model feature."""
from pathlib import Path
from hashlib import sha256
import json

import numpy as np
import pandas as pd

ULTRA_BEAST_IDS = frozenset((*range(793, 800), *range(803, 807)))
PARADOX_IDS = frozenset((*range(984, 996), 1005, 1006, 1009, 1010, 1020, 1021, 1022, 1023))
ULTRA_BEAST_NAMES = frozenset(('nihilego', 'buzzwole', 'pheromosa', 'xurkitree', 'celesteela',
                              'kartana', 'guzzlord', 'poipole', 'naganadel', 'stakataka', 'blacephalon'))
PARADOX_NAMES = frozenset(('great_tusk', 'scream_tail', 'brute_bonnet', 'flutter_mane',
                         'slither_wing', 'sandy_shocks', 'iron_treads', 'iron_bundle',
                         'iron_hands', 'iron_jugulis', 'iron_moth', 'iron_thorns',
                         'roaring_moon', 'iron_valiant', 'walking_wake', 'iron_leaves',
                         'gouging_fire', 'raging_bolt', 'iron_boulder', 'iron_crown'))
SCOPES = ('all', 'ordinary', 'ultra_beast', 'paradox')


def design_groups(rows):
    """Use species identity; exact names support older saved prediction tables."""
    result = pd.Series('ordinary', index=rows.index)
    if 'species_id' in rows:
        ultra = rows.species_id.isin(ULTRA_BEAST_IDS)
        paradox = rows.species_id.isin(PARADOX_IDS)
    else:
        names = rows.name.str.replace('-', '_', regex=False)
        ultra, paradox = names.isin(ULTRA_BEAST_NAMES), names.isin(PARADOX_NAMES)
    result.loc[ultra] = 'ultra_beast'
    result.loc[paradox] = 'paradox'
    return result


def selection_mask(rows, scope='ordinary'):
    if scope not in ('ordinary', 'all'):
        raise ValueError('selection_scope must be ordinary or all')
    return (design_groups(rows).eq('ordinary') if scope == 'ordinary'
            else pd.Series(True, index=rows.index))


def grouped_metrics(predictions, *, by_stage=False):
    """Score saved predictions without fitting; empty groups have no error score."""
    keys = ['population', 'generation', 'model']
    if by_stage:
        keys += ['rarity', 'stage']
    records = []
    for identity, rows in predictions.groupby(keys, observed=True, sort=True):
        groups = design_groups(rows)
        for scope in SCOPES:
            subset = rows if scope == 'all' else rows.loc[groups.eq(scope)]
            error = subset.actual_bst.to_numpy() - subset.pred_bst.to_numpy()
            record = dict(zip(keys, identity))
            record.update(evaluation_scope=scope, n_validation=len(subset),
                          MAE=float(np.abs(error).mean()) if len(error) else np.nan,
                          RMSE=float(np.sqrt((error ** 2).mean())) if len(error) else np.nan,
                          bias=float(error.mean()) if len(error) else np.nan)
            records.append(record)
    return pd.DataFrame(records)


def pooled_group_summary(metrics):
    from pokeml.evaluation.forward import pooled_summary
    positive = metrics.loc[metrics.n_validation > 0]
    result = pooled_summary(positive)
    bias = positive.copy()
    bias['bias_sum'] = bias.bias * bias.n_validation
    keys = ['population', 'model', 'evaluation_scope']
    aggregate = bias.groupby(keys, observed=True).agg(bias_sum=('bias_sum', 'sum'),
                                                     count=('n_validation', 'sum')).reset_index()
    aggregate['bias'] = aggregate.bias_sum / aggregate['count']
    return result.merge(aggregate[keys + ['bias']], on=keys, validate='one_to_one')


def save_historical_group_report(run_directory, output_root='artifacts/evaluation/special_groups'):
    """Write a separate, reproducible breakdown; never change old run selection."""
    parent = Path(run_directory)
    details = parent / 'development_details.csv'
    manifest_path = next(parent.glob('*_model_manifest.json'))
    from pokeml.pipeline.workflow import package_source_digest
    identity = dict(parent_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
                    predictions_sha256=sha256(details.read_bytes()).hexdigest(),
                    source_sha256=package_source_digest(),
                    note='New breakdown of old predictions; original winner remains unchanged.')
    digest = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(output_root) / f'{parent.name}_{digest}'
    if directory.exists():
        if json.loads((directory / 'manifest.json').read_text()) != identity:
            raise ValueError('Historical group report does not match its inputs')
        return directory
    rows = pd.read_csv(details)
    metrics = grouped_metrics(rows)
    directory.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(directory / 'evaluation_metrics.csv', index=False)
    pooled_group_summary(metrics).to_csv(directory / 'evaluation_summary.csv', index=False)
    grouped_metrics(rows, by_stage=True).to_csv(directory / 'evaluation_stage_errors.csv', index=False)
    (directory / 'manifest.json').write_text(json.dumps(identity, indent=2))
    return directory
