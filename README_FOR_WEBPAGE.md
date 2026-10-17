<!-- SHARED:START -->
# PokéML

Python · pandas · scikit-learn · CatBoost · LightGBM · XGBoost · Jupyter
<!-- SHARED:END -->

<!-- WEBPAGE:START -->
Can Pokémon attributes predict base stat total:
the sum of six base stats?

This project covers data preparation, exploratory analysis,
feature engineering, model comparison and predictions.
It compares direct regression with a baseline-plus-correction
approach using CatBoost, LightGBM and XGBoost.

Models train on earlier generations and are evaluated on later
generations. Target-derived rules learn from training data only.

Results remain retrospective. All validation generations guide
model selection, so no untouched final test exists. Predictions
also depend on supplied evolution-stage and rarity assumptions.
<!-- WEBPAGE:END -->

<!-- PANOPEPINO:START -->
A machine learning pipeline for predicting Pokémon base stat
totals. Compares gradient-boosting models using validation
across generations. Results are retrospective, with no untouched
final test.
<!-- PANOPEPINO:END -->
