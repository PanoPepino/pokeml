<!-- SHARED:START -->
# PokéML

Python · pandas · scikit-learn · CatBoost · LightGBM · XGBoost · Jupyter
<!-- SHARED:END -->

<!-- WEBPAGE:START -->
The idea of this project is simple: Can next generation Pokémon stats be 
predicted by training ML with current Pokédex data (1082 Pokémon + variants)?

This project works out the end-to end process. From data preparation, 
exploratory analysis, feature engineering, to model comparison and predictions.
It compares direct regression with a baseline-plus-correction
approach using CatBoost, LightGBM and XGBoost.

Results remain retrospective. All validation generations guide
model selection, so no untouched final test exists. Predictions
also depend on supplied evolution-stage and rarity assumptions.
<!-- WEBPAGE:END -->

<!-- PANOPEPINO:START -->
This is a machine learning pipeline for predicting next generation
Pokémon stats. It compares gradient-boosting models using validation
across generations. Results are retrospective, with no untouched final test.
<!-- PANOPEPINO:END -->
