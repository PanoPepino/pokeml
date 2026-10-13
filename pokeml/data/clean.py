"""
Script to clean and arrange the complete scrapped Pkdx
"""


import pandas as pd

from pathlib import Path
from pokeml.constants import REGIONS, BEAST_GEN_9, PARADOX, PARADOX_LEGEND, MINIMAL_FEATURES, MINIMAL_METADATA
from pokeml.data.load import load_data
from pokeml.data.evolution import EvolutionReference
from pokeml.data.validation import validate_pkdx




def parse_and_rename(df):
    """
    Rename pokes for easier manipulation.
    """
    df = df.copy()

    # 1) Normalize separators
    # Reference identifies default forms explicitly. Never truncate a name to
    # its first token: that loses Jangmo-o and regional/form identity.
    names = df["canonical_name"] if "canonical_name" in df else df["name"]
    df["name"] = names.str.replace("-", "_", regex=False)

    # 5) Category normalization -> rarity
    df["category"] = df["category"].replace({
        "mythical": "legendary",
        "baby": "regular",
    })
    df = df.rename(columns={"category": "rarity"})

    return df


def update_generation(df):
    """
    Update target_column where search_column contains word. When a given string (in words) is detected, the generation in gens get updated.
    """

    df = df.copy()
    gens = ['7', '8', '8', '9']
    for word, gen in zip(REGIONS, gens):
        mask = df['name'].str.contains(rf"(?:^|_){word}(?:_|$)", case=False, na=False)
        df.loc[mask, 'generation'] = int(gen)
    df.loc[df['name'].isin(['dialga_origin', 'palkia_origin']), 'generation'] = 8
    return df


def change_stage(df):
    """
    Fix stage and rarity values for special cases.
    """
    df = df.copy()

    df.loc[df["name"].isin(PARADOX), "stage"] = "single"
    df.loc[df["name"].isin(PARADOX_LEGEND), "rarity"] = "legendary"
    df.loc[df["name"] == "meltan", "stage"] = "s1c2"
    df.loc[df["name"] == "melmetal", "stage"] = "s2c2"
    df.loc[df["name"].isin(BEAST_GEN_9), "stage"] = "single"

    return df


def split_type_egg(df):
    """
    Simply to split types and eggs.
    """
    df = df.copy()

    # Reindex supports datasets where every Pokémon has only one type/group.
    df[["type_1", "type_2"]] = df["types"].str.split(r"\s*\|\s*", n=1, expand=True).reindex(columns=[0, 1])
    df[["egg_group_1", "egg_group_2"]] = df["egg_groups"].str.split(r"\s*\|\s*", n=1, expand=True).reindex(columns=[0, 1])

    df = df.drop(columns=["types", "egg_groups"])

    return df


def clean_pkdx_raw(in_path: Path | str,
                   out_path: Path | str,
                   reference_dir=None) -> pd.DataFrame:
    """Clean local raw records with pinned identity/evolution metadata."""

    path_to_pd = Path(in_path)
    out_pd = Path(out_path)
    if path_to_pd.resolve() == out_pd.resolve():
        raise ValueError('Raw and clean paths must differ')

    raw_df = load_data(path_to_pd)

    reference = EvolutionReference() if reference_dir is None else EvolutionReference(reference_dir)
    annotated = reference.annotate(raw_df)
    parname = parse_and_rename(annotated)
    upgen = update_generation(parname)
    chstg = change_stage(upgen)
    type_egg = split_type_egg(chstg)
    type_egg = type_egg.drop(columns=["canonical_name"])
    validate_pkdx(type_egg, full=True)

    out_pd.parent.mkdir(parents=True, exist_ok=True)
    type_egg.to_csv(out_pd, index=False)
    return type_egg


def get_pkdx_minimal(
        in_path: Path | str,
        out_path: Path | str,
        features: list[str] | None = None) -> pd.DataFrame:
    """Select analysis fields plus identifiers needed for temporal/family checks.

    Explicit feature lists remain supported. Metadata never becomes a predictor.
    """

    path_to_pd = Path(in_path)
    out_pd = Path(out_path)
    if path_to_pd.resolve() == out_pd.resolve():
        raise ValueError('Clean and minimal paths must differ')

    df = load_data(path_to_pd)
    validate_pkdx(df, full=True)
    columns = list(dict.fromkeys([*(features if features is not None else MINIMAL_FEATURES), *MINIMAL_METADATA]))
    trimmed_pkdx = df[columns].copy()
    validate_pkdx(trimmed_pkdx)

    out_pd.parent.mkdir(parents=True, exist_ok=True)
    trimmed_pkdx.to_csv(out_pd, index=False)
    return trimmed_pkdx
