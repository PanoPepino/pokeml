"""Reproducible raw -> clean -> minimal build with provenance and corrections."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pokeml.data.clean import clean_pkdx_raw, get_pkdx_minimal
from pokeml.data.evolution import EvolutionReference
from pokeml.data.validation import validate_pkdx


def build_pkdx(input_path="datasets/pkdx_raw.csv", output_path_clean="datasets/pkdx_clean.csv",
               output_path_minimal="datasets/pkdx_min.csv", reference_dir=None):
    """Build analysis snapshot offline; raw measurements remain unchanged.

    Returns clean/minimal frames and writes a manifest and correction table
    beside minimal CSV. Current evolution metadata is not a historical snapshot.
    """
    input_path, output_path_clean, output_path_minimal = map(
        Path, (input_path, output_path_clean, output_path_minimal)
    )
    paths = [p.resolve() for p in (input_path, output_path_clean, output_path_minimal)]
    if len(set(paths)) != 3:
        raise ValueError("Raw, clean and minimal paths must differ")
    raw = pd.read_csv(input_path)
    reference = EvolutionReference() if reference_dir is None else EvolutionReference(reference_dir)
    clean = clean_pkdx_raw(input_path, output_path_clean, reference_dir)
    minimal = get_pkdx_minimal(output_path_clean, output_path_minimal)
    corrections = pd.DataFrame({
        'id': raw.id, 'raw_name': raw.name, 'name': clean.name,
        'raw_stage': raw.stage, 'stage': clean.stage,
        'raw_generation': raw.generation, 'generation': clean.generation,
        'stage_source': clean.stage_source,
    })
    changed = ((corrections.raw_name != corrections.name) |
               (corrections.raw_stage != corrections.stage) |
               (corrections.raw_generation != corrections.generation))
    corrections.loc[changed].to_csv(output_path_minimal.with_name('pkdx_corrections.csv'), index=False)
    manifest = {
        'built_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Existing raw snapshot; selected regional/origin forms, not every alternate form',
        'evolution_metadata': 'Current known branches; includes later-generation evolutions',
        'stage_policy': 'Branch-specific length; ambiguous ancestors retain longest eligible route',
        'rarity_policy': 'Mythical merged into legendary; babies into regular; existing Paradox overrides retained',
        'units': {'height': 'metres', 'weight': 'kilograms', 'total_stats': 'sum of six base stats'},
        'reference': reference.manifest,
        'validation': validate_pkdx(minimal),
        'stage_corrections_vs_raw': int((raw.stage != clean.stage).sum()),
        'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (input_path, output_path_clean, output_path_minimal)},
        'minimal_columns': minimal.columns.tolist(),
        'excluded_from_predictors': ['id', 'species_id', 'evolution_chain_id', 'generation'],
    }
    output_path_minimal.with_name('pkdx_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return clean, minimal
