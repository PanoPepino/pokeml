

# List of pokemon name exceptions to not die when parsing
REGULAR_POKES = ['nidoran_f',
                 'nidoran_m',
                 'mr_mime',
                 'ho_oh',
                 'mime_jr',
                 'porygon_z',
                 'type_null',
                 'jangmo_o',
                 'hakamo_o',
                 'kommo_o',
                 'tapu_koko',
                 'tapu_lele',
                 'tapu_bulu',
                 'tapu_fini',
                 'mr_rime',
                 'great_tusk',
                 'scream_tail',
                 'brute_bonnet',
                 'flutter_mane',
                 'slither_wing',
                 'sandy_shocks',
                 'iron_treads',
                 'iron_bundle',
                 'iron_hands',
                 'iron_jugulis',
                 'iron_moth',
                 'iron_thorns',
                 'wo_chien',
                 'chien_pao',
                 'ting_lu',
                 'chi_yu',
                 'roaring_moon',
                 'iron_valiant',
                 'walking_wake',
                 'iron_leaves',
                 'gouging_fire',
                 'raging_bolt',
                 'iron_boulder',
                 'iron_crown',
                 '_alola',
                 '_galar',
                 '_hisui',
                 '_paldea',
                 ]


INITIAL_POKES = [
    "bulbasaur", "ivysaur", "venusaur",
    "charmander", "charmeleon", "charizard",
    "squirtle", "wartortle", "blastoise",
    "chikorita", "bayleef", "meganium",
    "cyndaquil", "quilava", "typhlosion",
    "totodile", "croconaw", "feraligatr",
    "treecko", "grovyle", "sceptile",
    "torchic", "combusken", "blaziken",
    "mudkip", "marshtomp", "swampert",
    "turtwig", "grotle", "torterra",
    "chimchar", "monferno", "infernape",
    "piplup", "prinplup", "empoleon",
    "snivy", "servine", "serperior",
    "tepig", "pignite", "emboar",
    "oshawott", "dewott", "samurott",
    "chespin", "quilladin", "chesnaught",
    "fennekin", "braixen", "delphox",
    "froakie", "frogadier", "greninja",
    "rowlet", "dartrix", "decidueye",
    "litten", "torracat", "incineroar",
    "popplio", "brionne", "primarina",
    "grookey", "thwackey", "rillaboom",
    "scorbunny", "raboot", "cinderace",
    "sobble", "drizzile", "inteleon",
    "sprigatito", "floragato", "meowscarada",
    "fuecoco", "crocalor", "skeledirge",
    "quaxly", "quaxwell", "quaquaval",
]


REGIONS = [
    "alola", "galar", "hisui", "paldea"
]


# Paradox Pokémon list (with underscores after name replacement)
PARADOX = ['great_tusk',
           'scream_tail',
           'brute_bonnet',
           'flutter_mane',
           'slither_wing',
           'sandy_shocks',
           'iron_treads',
           'iron_bundle',
           'iron_hands',
           'iron_jugulis',
           'iron_moth',
           'iron_thorns',
           'roaring_moon',
           'iron_valiant',
           'walking_wake',
           'iron_leaves',
           'gouging_fire',
           'raging_bolt',
           'iron_boulder',
           'iron_crown']


# paradox and beast of gen 9that are legendary:
PARADOX_LEGEND = ['walking_wake',
                  'gouging_fire',
                  'raging_bolt',
                  'iron_leaves',
                  'iron_crown',
                  'iron_boulder']

BEAST_GEN_9 = ['wo_chien',
               'chien_pao',
               'ting_lu',
               'chi_yu']


MINIMAL_FEATURES = [
    'name',
    'type_1',
    'type_2',
    'rarity',
    'stage',
    'shape',
    'color',
    'total_stats',
    'height',
    'weight']
