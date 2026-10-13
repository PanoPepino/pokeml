"""Shared model-input schema. Extra Pokédex statistics are never predictors."""

from pokeml.constants import MINIMAL_FEATURES

TARGET = "total_stats"
BASE_FEATURES = tuple(c for c in MINIMAL_FEATURES if c not in ("name", TARGET))
METADATA_COLUMNS = ("id", "generation", "species_id", "evolution_chain_id")


def model_frame(frame, engineered_columns=()):
    """Select only declared inputs; protect against raw target components leaking in."""
    columns = list(BASE_FEATURES) + list(engineered_columns)
    return frame.loc[:, list(dict.fromkeys(columns))].copy()
