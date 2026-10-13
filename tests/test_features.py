"""Regression checks for real pandas transforms; no model dependencies needed."""

import unittest
import numpy as np
import pandas as pd

from pokeml.data.load import load_data
from pokeml.data.catfill import cat_fill
from pokeml.features.feature_typing import add_dual_type_dev_features, add_mono_type_dev_features
from pokeml.features.feature_initial import apply_initial_tag, get_initial_pokes
from pokeml.pipeline.schema import model_frame


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "name": ["first", "mono", "last"], "rarity": ["regular"] * 3,
            "stage": ["single"] * 3, "type_1": ["fire", "water", "fire"],
            "type_2": ["water", None, "water"],
        }, index=[70, 12, 70])
        self.mapping = pd.DataFrame({
            "rarity": ["regular", "regular"], "stage": ["single", "single"],
            "type": ["fire", "water"], "dev_z_t1": [1.2, 5.0],
            "dev_z_t2": [2.0, -0.7], "dev_z_t1_isna": [0, 0], "dev_z_t2_isna": [0, 0],
        })

    def test_dual_join_preserves_sparse_duplicate_indices(self):
        result = add_dual_type_dev_features(self.frame, self.mapping)
        self.assertEqual(list(result.index), list(self.frame.index))
        self.assertEqual(list(result.name), list(self.frame.name))
        np.testing.assert_allclose(result.dev_z_t1, [1.2, np.nan, 1.2], equal_nan=True)
        np.testing.assert_allclose(result.dev_z_t2, [-0.7, np.nan, -0.7], equal_nan=True)

    def test_mono_join_preserves_identity(self):
        mapping = pd.DataFrame({"rarity": ["regular"], "stage": ["single"],
                                "type_1": ["water"], "mono_dev_z": [0.3],
                                "mono_dev_z_isna": [0]})
        result = add_mono_type_dev_features(self.frame, mapping)
        self.assertEqual(list(result.index), [70, 12, 70])
        np.testing.assert_allclose(result.mono_dev_z, [np.nan, .3, np.nan], equal_nan=True)

    def test_load_data_is_nonmutating(self):
        copy = load_data(self.frame)
        copy.loc[:, "name"] = "changed"
        self.assertEqual(self.frame.name.iloc[0], "first")

    def test_string_categorical_fill(self):
        frame = self.frame.astype({"type_2": "string"})
        self.assertEqual(cat_fill(frame).type_2.iloc[1], "None")
        self.assertTrue(pd.isna(frame.type_2.iloc[1]))

    def test_new_and_regional_starters(self):
        data = pd.DataFrame({"name": ["browt", "pombon", "gecqua", "typhlosion_hisui", "other"]})
        tagged = apply_initial_tag(data, get_initial_pokes())
        self.assertEqual(tagged.initial.astype(str).tolist(),
                         ["initial", "initial", "initial", "initial", "non_initial"])

    def test_target_components_and_metadata_excluded(self):
        from pokeml.pipeline.schema import BASE_FEATURES
        frame = pd.DataFrame({name: [1] for name in BASE_FEATURES})
        frame["hp"] = 100
        frame["total_stats"] = 600
        frame["generation"] = 9
        self.assertEqual(list(model_frame(frame)), list(BASE_FEATURES))


if __name__ == "__main__":
    unittest.main()
