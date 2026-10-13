"""Identity, branch semantics and offline reproducibility of analysis snapshot."""

from pathlib import Path

import pandas as pd
import pytest

from pokeml.data.build import build_pkdx
from pokeml.data.clean import split_type_egg
from pokeml.data.evolution import EvolutionReference
from pokeml.data.validation import validate_pkdx
from pokeml.pipeline.schema import model_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,stage', [
    ('jangmo-o', 's1c3'), ('hakamo-o', 's2c3'), ('kommo-o', 's3c3'),
    ('nidoran-f', 's1c3'), ('nidoran-m', 's1c3'), ('porygon-z', 's3c3'),
    ('type-null', 's1c2'), ('mime-jr', 's1c3'), ('mr-mime', 's2c2'),
    ('mr-mime-galar', 's2c3'), ('mr-rime', 's3c3'),
    ('farfetchd', 'single'), ('farfetchd-galar', 's1c2'),
    ('corsola', 'single'), ('corsola-galar', 's1c2'),
    ('qwilfish', 'single'), ('qwilfish-hisui', 's1c2'),
    ('zigzagoon', 's1c2'), ('linoone', 's2c2'), ('linoone-galar', 's2c3'),
    ('flapple', 's2c2'), ('appletun', 's2c2'), ('dipplin', 's2c3'),
    ('hydrapple', 's3c3'), ('basculin-red-striped', 'single'),
    ('phione', 'single'), ('manaphy', 'single'), ('meltan', 's1c2'), ('melmetal', 's2c2'),
])
def test_stage_routes(name, stage):
    assert EvolutionReference().resolve(name)['stage'] == stage


def test_unknown_name_fails():
    with pytest.raises(ValueError, match='Unknown'):
        EvolutionReference().resolve('not-a-pokemon')


def test_acquisition_stage_lookup_is_offline_and_preserves_hyphens(monkeypatch):
    from pokeml.data import acquire
    def reject_network(*args, **kwargs):
        raise AssertionError('Known stage should resolve from local reference')
    monkeypatch.setattr(acquire, '_get_json', reject_network)
    assert acquire.get_stage('jangmo-o') == 's1c3'
    assert acquire.get_stage('porygon-z') == 's3c3'
    assert acquire.get_stage('mr-mime-galar') == 's2c3'


def test_reference_checksum_is_enforced(tmp_path):
    import shutil
    source = ROOT / 'datasets/reference/pokeapi'
    for name in ['pokemon_species.csv', 'pokemon.csv', 'manifest.json']:
        shutil.copy(source / name, tmp_path / name)
    with (tmp_path / 'pokemon.csv').open('a') as output:
        output.write('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        EvolutionReference(tmp_path)


def test_single_type_and_egg_group():
    frame = pd.DataFrame({'types': ['fire'], 'egg_groups': ['monster']})
    result = split_type_egg(frame)
    assert result.type_1.iloc[0] == 'fire'
    assert result.type_2.isna().all()
    assert result.egg_group_2.isna().all()


def test_offline_build_preserves_stats_and_source(tmp_path, monkeypatch):
    import requests
    def reject_network(*args, **kwargs):
        raise AssertionError('Offline build attempted network access')
    monkeypatch.setattr(requests, 'get', reject_network)
    source = ROOT / 'datasets/pkdx_raw.csv'
    before = source.read_bytes()
    raw = pd.read_csv(source)
    clean, minimal = build_pkdx(source, tmp_path / 'clean.csv', tmp_path / 'minimal.csv')
    assert source.read_bytes() == before
    assert len(clean) == len(minimal) == len(raw) == 1082
    for column in ['id', 'height', 'weight', 'total_stats', 'hp', 'atk', 'def_', 'sp_atk', 'sp_def', 'spe']:
        pd.testing.assert_series_equal(raw[column], clean[column])
    assert clean.name.is_unique and clean.id.is_unique
    by_name = clean.set_index('name')
    assert by_name.loc['darmanitan_galar_zen', 'generation'] == 8
    assert by_name.loc['darmanitan_galar_zen', 'total_stats'] == 540
    assert by_name.loc['typhlosion_hisui', 'generation'] == 8
    assert by_name.loc['dialga_origin', 'generation'] == 8
    assert set(minimal.columns) - set(model_frame(minimal).columns) == {
        'name', 'total_stats', 'id', 'species_id', 'evolution_chain_id', 'generation'}
    saved = pd.read_csv(ROOT / 'datasets/pkdx_min.csv')
    pd.testing.assert_frame_equal(minimal, saved)


def test_validation_rejects_duplicate_and_incorrect_bst():
    frame = pd.read_csv(ROOT / 'datasets/pkdx_clean.csv')
    with pytest.raises(ValueError, match='Duplicate'):
        validate_pkdx(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, 'hp'] += 1
    with pytest.raises(ValueError, match='sum'):
        validate_pkdx(frame, full=True)


def test_no_raw_overwrite(tmp_path):
    with pytest.raises(ValueError, match='must differ'):
        build_pkdx(tmp_path / 'raw.csv', tmp_path / 'raw.csv', tmp_path / 'min.csv')
