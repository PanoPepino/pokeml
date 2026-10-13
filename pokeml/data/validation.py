"""Input schema checks without presentation dependencies."""
import re

def validate_required_columns(df, required_cols, func_name="function"):
    """
    To validate columns appearing when checking dual type ordering deviations.
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise KeyError(
            f"{func_name}: Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_pkdx(df, full=False):
    """Fail before writing an incomplete, ambiguous or inconsistent Pokédex."""
    from pokeml.constants import MINIMAL_FEATURES, MINIMAL_METADATA
    import numpy as np

    required = set(MINIMAL_FEATURES + MINIMAL_METADATA)
    stats = ['hp', 'atk', 'def_', 'sp_atk', 'sp_def', 'spe']
    if full:
        required.update(stats)
    validate_required_columns(df, required, 'validate_pkdx')
    if df.empty:
        raise ValueError('Pokédex must contain records')
    if df[list(required - {'type_2'})].isna().any().any():
        raise ValueError('Missing required Pokédex values (only secondary type may be absent)')
    for key in ['id', 'name']:
        if df[key].duplicated().any():
            raise ValueError(f'Duplicate Pokémon {key}')
    for key in ['id', 'species_id', 'evolution_chain_id', 'generation', 'total_stats']:
        values = df[key].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
            raise ValueError(f'Invalid positive integer field: {key}')
    for key in ['height', 'weight']:
        values = df[key].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f'Invalid positive measurement: {key}')
    for stage in df['stage']:
        match = re.fullmatch(r's([1-3])c([2-3])', stage)
        if stage != 'single' and (match is None or int(match[1]) > int(match[2])):
            raise ValueError(f'Invalid evolution stage: {stage}')
    if full and not (df[stats].sum(axis=1) == df['total_stats']).all():
        raise ValueError('BST differs from sum of six base stats')
    types = {'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
             'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'}
    if not set(df['type_1']).issubset(types) or not set(df['type_2'].dropna()).issubset(types):
        raise ValueError('Unknown Pokémon type')
    if not set(df['rarity']).issubset({'regular', 'legendary'}):
        raise ValueError('Unknown rarity')
    return {'rows': len(df), 'species': int(df['species_id'].nunique()),
            'families': int(df['evolution_chain_id'].nunique()), 'valid': True}
