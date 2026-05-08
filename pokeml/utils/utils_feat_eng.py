from pokeml.features.feature_registry import get_feature_steps
import typer


def resolve_feat_steps(feat_mode: str, feat_steps: str):
    if feat_mode == "none":
        return []
    if feat_mode == "full":
        return get_feature_steps()
    if feat_mode == "custom":
        names = [x.strip() for x in feat_steps.split(",") if x.strip()]
        return get_feature_steps(active_steps=names)
    raise typer.BadParameter("feat_mode must be: none, full, custom")
