# Data preparation

## Source and reproducibility

`datasets/pkdx_raw.csv` is the pinned source snapshot. Current clean/minimal
tables and metadata are retained; obsolete backup snapshots were removed.
No Pokémon stats were refreshed and no images were downloaded.

Two official PokeAPI repository CSVs were downloaded for identity and evolution
metadata. `datasets/reference/pokeapi/manifest.json` records commit, URLs, retrieval
time and SHA256 hashes. Builds verify these hashes and operate offline.

```bash
pokeml data clean
```

Python entrypoint: `pokeml.data.build.build_pkdx(...)`.
It writes validated clean/minimal CSVs, `pkdx_manifest.json` and
`pkdx_corrections.csv`. Input and output paths must differ. Acquisition remains
separate; metadata lookup failures never become `single` silently.

## Minimal analysis table

`datasets/pkdx_min.csv` contains 1,082 rows: 1,025 species plus 57 selected forms.
It is not a census of every alternate form.

| Fields | Meaning |
|---|---|
| `name` | Unique canonical name; meaningful form suffixes retained |
| `type_1`, `type_2` | Original type slots; missing secondary type means monotype |
| `rarity`, `stage` | Project rarity policy and current evolution route |
| `shape`, `color` | Original snapshot labels; prerelease availability is not guaranteed |
| `height`, `weight` | Metres and kilograms |
| `total_stats` | Sum of six base stats; prediction target |
| `id` | Original PokeAPI Pokémon/form ID; no sequential renumbering |
| `species_id` | Shared species ID across forms |
| `evolution_chain_id` | Family identifier for dependence/leakage checks |
| `generation` | Species or specific form's introduction generation |

IDs and generation are analysis metadata. Model schema excludes them, name,
target and all component stats from predictors.

## Evolution and form policy

Full identifiers resolve through Pokémon -> species. Hyphens are not blindly
stripped. Nidoran, Jangmo-o/Hakamo-o/Kommo-o, Porygon-Z, Type: Null and Mime
labels now resolve correctly.

Stage uses position from root plus remaining depth within that species' branch.
Flapple/Appletun are `s2c2`; Dipplin is `s2c3`. Ambiguous ancestors retain longest
possible route: Applin `s1c3`, Mime Jr. `s1c3`. Stage labels alone therefore do
not describe every possible descendant route.

Explicit form routes distinguish ordinary Farfetch'd/Corsola/Qwilfish/Basculin
from forms that evolve, ordinary Zigzagoon/Linoone from Galarian chain, and
ordinary Mr. Mime from Galarian Mr. Mime. Phione/Manaphy are single-stage;
breeding relationship is not an evolution. Meltan/Melmetal remain a two-stage chain.

Galarian Zen Darmanitan retains its own name, original ID and BST. Origin Dialga
and Palkia retain `origin`, rather than being renamed `hisui`. Hisuian and these
Origin forms are generation 8. Other regional forms use their introduction generation.

These labels describe **current known chains**, including evolutions introduced
after ancestors. They are not historical prerelease information. Community stage
guesses and announcement-time evaluation still require separate handling.

Existing rarity reasoning is preserved: mythical -> legendary, baby -> regular,
plus existing overrides for six Paradox Pokémon. Clean table retains original
legendary/mythical flags; project rarity is not identical to official flags.

## Validation and exploration

Build validates unique names/IDs, required fields, positive finite measurements,
integer metadata/BST, stage syntax, known types/rarity, and BST equal to component
sum. Tests compare rebuilt minimal table with checked-in table, prohibit network
calls during build, and check raw measurements remain unchanged.

Start `notebooks/00_data_preparation.ipynb`. It displays source hashes, corrections,
missingness, stage/rarity counts, generation counts and stage BST distributions.
Continue with stage, power-creep and typing notebooks in numbered order. Original
comments are preserved beside revised analysis. Old fitted models/results need
regeneration after dataset corrections.
