from pokeml.constants import INITIAL_POKES


def get_initial_pokes(df=None):
    return {
        "initial_pokes": INITIAL_POKES,
        "new_cat_feats": ["initial"],
        "new_maps": None,
    }


def apply_initial_tag(df, state):
    out = df.copy()
    out["initial"] = out["name"].isin(state["initial_pokes"]).map(
        {True: "initial", False: "non_initial"}
    ).astype("category")
    return out
