from pokeml.constants import INITIAL_POKES


def get_initial_pokes(df=None):
    return {
        "initial_pokes": INITIAL_POKES,
        "new_cat_feats": ["initial"],
        "new_maps": None,
    }


def apply_initial_tag(df, state):
    out = df.copy()
    out["initial"] = out["name"].str.replace(r"_(alola|galar|hisui|paldea)$", "", regex=True).isin(
        state["initial_pokes"] + ["browt", "pombon", "gecqua"]
    ).map(
        {True: "initial", False: "non_initial"}
    ).astype("category")
    return out
