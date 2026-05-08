from pokeml.features.feature_initial import get_initial_pokes, apply_initial_tag
from pokeml.features.feature_typing import get_type_deviation_state, apply_type_deviation_features

#  Registry ------------------------

FEATURE_STEPS = [
    ("initial_tag", get_initial_pokes, apply_initial_tag),
    ("type_deviation", get_type_deviation_state, apply_type_deviation_features)
]


def get_feature_steps(mode="full", active_steps=None):
    if mode == "none":
        return []
    if mode == "full":
        return FEATURE_STEPS
    if mode == "custom":
        wanted = set(active_steps or [])
        return [step for step in FEATURE_STEPS if step[0] in wanted]
    raise ValueError(f"Unknown mode: {mode}")
