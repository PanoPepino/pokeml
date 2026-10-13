"""Offline species identity and evolution labels from pinned PokeAPI tables.

Stages describe current known evolution routes, not knowledge at announcement.
For branching trees, chain length follows the branch containing the species.
Regional-only evolutions need form-specific routes because species tables merge
ordinary and regional forms into one tree.
"""

import csv
import hashlib
import json
from pathlib import Path

REFERENCE_DIRECTORY = Path(__file__).resolve().parents[2] / "datasets/reference/pokeapi"

# Species-level trees also contain evolutions exclusive to another form.
# These ordinary forms cannot take that route.
FORM_STAGES = {
    "farfetchd": "single", "farfetchd-galar": "s1c2",
    "corsola": "single", "corsola-galar": "s1c2",
    "qwilfish": "single", "qwilfish-hisui": "s1c2",
    "zigzagoon": "s1c2", "linoone": "s2c2",
    "zigzagoon-galar": "s1c3", "linoone-galar": "s2c3",
    "mr-mime": "s2c2", "mr-mime-galar": "s2c3",
    # Mime Jr. can precede either Mr. Mime route. Retain longest possible route.
    "mime-jr": "s1c3",
    "basculin-red-striped": "single", "basculin-blue-striped": "single",
    "basculin-white-striped": "s1c2",
}


class EvolutionReference:
    """Resolve full Pokémon identifiers without stripping meaningful hyphens."""

    def __init__(self, directory=REFERENCE_DIRECTORY):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        for filename in ("pokemon_species.csv", "pokemon.csv"):
            digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            if digest != self.manifest["files"][filename]["sha256"]:
                raise ValueError(f"Reference checksum mismatch: {filename}")
        with (directory / "pokemon_species.csv").open(newline="") as source:
            self.species = {int(row["id"]): row for row in csv.DictReader(source)}
        with (directory / "pokemon.csv").open(newline="") as source:
            self.pokemon = {int(row["id"]): row for row in csv.DictReader(source)}
        self.by_name = {row["identifier"]: row for row in self.pokemon.values()}
        self.children = {identifier: [] for identifier in self.species}
        for identifier, row in self.species.items():
            if row["evolves_from_species_id"]:
                self.children[int(row["evolves_from_species_id"])].append(identifier)

    def _depth(self, identifier, seen=()):
        if identifier in seen:
            raise ValueError("Cycle in evolution reference")
        parent = self.species[identifier]["evolves_from_species_id"]
        return 1 if not parent else 1 + self._depth(int(parent), (*seen, identifier))

    def _remaining(self, identifier, seen=()):
        if identifier in seen:
            raise ValueError("Cycle in evolution reference")
        children = self.children[identifier]
        return 1 if not children else 1 + max(
            self._remaining(child, (*seen, identifier)) for child in children
        )

    def resolve(self, identifier):
        """Return identity, family and route labels; unknown identifiers raise."""
        key = str(identifier).replace("_", "-")
        pokemon = self.pokemon.get(int(key)) if key.isdigit() else self.by_name.get(key)
        if pokemon is None:
            raise ValueError(f"Unknown Pokémon identifier: {identifier}")
        species_id = int(pokemon["species_id"])
        species = self.species[species_id]
        name = pokemon["identifier"]
        depth = self._depth(species_id)
        length = depth + self._remaining(species_id) - 1
        stage = "single" if length == 1 else f"s{depth}c{length}"
        stage = FORM_STAGES.get(name, stage)
        generation = int(species["generation_id"])
        for region, introduced in (("alola", 7), ("galar", 8), ("hisui", 8), ("paldea", 9)):
            if region in name.split("-"):
                generation = introduced
        if name in ("dialga-origin", "palkia-origin"):
            generation = 8  # Origin Formes introduced in Legends: Arceus.
        return {
            "source_name": name,
            "species_id": species_id,
            "evolution_chain_id": int(species["evolution_chain_id"]),
            "species_generation": int(species["generation_id"]),
            "generation": generation,
            "stage": stage,
            "stage_source": "form_route" if name in FORM_STAGES else "species_branch",
            "canonical_name": (species["identifier"] if pokemon["is_default"] == "1" else name).replace("-", "_"),
        }

    def annotate(self, frame):
        """Attach reference metadata without changing stats or row order."""
        import pandas as pd

        out = frame.copy()
        records = []
        for row in out.itertuples():
            record = self.resolve(row.id)
            if row.name.replace("_", "-") != record["source_name"]:
                raise ValueError(f"Pokémon id/name mismatch: {row.id}, {row.name}")
            records.append(record)
        annotations = pd.DataFrame(records)
        for column in annotations:
            out[column] = annotations[column].to_numpy()
        return out
