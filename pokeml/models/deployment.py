"""Versioned offline predictor; loading and serving never fit or acquire data."""

import json
from pathlib import Path
import joblib
import numpy as np

from pokeml.models.baseline import StageMedianBaseline
from pokeml.models.residual import BaselineCorrectionRegressor

FORMAT_VERSION = 1


class PokemonPredictor:
    """One deployable interface shared by notebooks, CSV commands and future APIs."""

    def __init__(self, model, model_version, metadata=None):
        self.model = model
        self.model_version = model_version
        self.metadata = metadata or {}

    def save(self, release_directory):
        directory = Path(release_directory)
        directory.mkdir(parents=True, exist_ok=False)
        joblib.dump(self.model, directory / "predictor.joblib", compress=3)
        manifest = {**self.metadata, "format_version": FORMAT_VERSION,
                    "model_version": self.model_version,
                    "required_metadata": ["name", "stage", "rarity"],
                    "model_kind": type(self.model).__name__,
                    "prediction_mode": getattr(self.model, "mode", "baseline"),
                    "feature_schema": list(getattr(self.model, "feature_names_in_", [])),
                    "uncertainty_note": "Conditional model estimates exclude stage and baseline-estimation uncertainty."}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return directory

    @classmethod
    def load(cls, release_directory):
        directory = Path(release_directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("Incompatible release format; legacy direct artifacts are retired; create a versioned release with model workflow")
        model = joblib.load(directory / "predictor.joblib")
        if not isinstance(model, (StageMedianBaseline, BaselineCorrectionRegressor)):
            raise ValueError("Release does not contain a supported predictor")
        return cls(model, manifest["model_version"], manifest)

    def predict(self, rows):
        if "name" not in rows or rows.name.isna().any():
            raise ValueError("Prediction requires nonmissing name column")
        result = self.model.predict_components(rows).copy()
        distribution = self.model.predict_distribution(rows)
        if not np.allclose(result.pred_bst, distribution["pred_bst"]):
            raise ValueError("Point and distribution predictions disagree")
        for column, values in distribution.items():
            if column != "pred_bst":
                result[column] = values
        result.insert(0, "name", rows.name.to_numpy())
        result.insert(1, "stage", rows.stage.to_numpy())
        result["model_version"] = self.model_version
        return result
