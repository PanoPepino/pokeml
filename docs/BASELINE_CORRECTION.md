# Direct BST and baseline + correction

## Two methods, four models

Direct method learns `BST = model(attributes)`.
Correction method learns `BST = training baseline + model(attributes)`.
Training correction target is actual BST minus baseline for that training Pokémon.
Baseline is median BST by stage/rarity; buckets below five rows fall back to rarity
median, then global median. Median-only prediction is separate reference model.

Both methods compare CatBoost native, CatBoost ordinal, LightGBM and XGBoost.
Inputs are type slots, stage, rarity, shape, colour, height, weight and starter tag.
Each fold learns its own baseline/features. No check labels fit transforms or
choose early stopping rounds. See [plain-language validation](VALIDATION.md).

## Train, tune and predict

```python
from pokeml.pipeline.workflow import WorkflowConfig, run_workflow
result = run_workflow(WorkflowConfig(run_name="my_unique_run", tune_iterations=60))
```

Zero tuning iterations uses starting settings. A positive budget samples that many
unique configurations per model/method, including defaults, with fixed seed.
Four chronological folds choose settings and model using pooled ordinary-Pokémon
MAE. RMSE breaks ties. `selection_scope="all"` uses all validation Pokémon.
LightGBM/XGBoost skip auxiliary quantile fitting during search, then restore it
for evaluation and final release. Search settings live in `configs/tuning_forward.json`.
No untouched final test exists. Final selected configuration trains on generations
1–9 including special groups; generation-9 evaluation models train through 8.

```bash
python -m pokeml.cli model workflow --run-name my_unique_run --tune-iterations 60
python -m pokeml.cli model predict-release --release-directory artifacts/workflow/my_unique_run/release --output-path artifacts/predictions/new_predictions.csv
```

Prediction receives announcement attributes, with supplied stage/rarity and height
in metres/weight in kilograms. Missing secondary type means monotype. Output
shows baseline, correction and predicted BST. For direct models, correction is
post-hoc prediction minus reference; it is not a residual training target.

## Saved results and diagnostics

Current saved run: `artifacts/workflow/pokemon_tuned_60_v1`.
Original all-Pokémon validation selected direct CatBoost native, MAE 30.87 over
433 Pokémon. New group breakdown gives ordinary MAE 28.52 over 402 Pokémon;
this reuses old predictions and does not change original winner.

Runs contain manifests, model settings, tuning trials, scores, row predictions,
group/stage reports, evaluation models and final release. New directories prevent
overwriting existing runs. Historical reporting never rewrites original results.

Training curves live in `artifacts/diagnostics/pokemon_tuned_60_curves_v2` and notebook
05. Curves are sampled tree prefixes from frozen settings; endpoints reproduce
saved scores. They include special groups. Low late deterioration does not prove
absence of overfitting or remove tuning-selection optimism.

## Uncertainty and known failures

CatBoost reports predictive sigma, not calibrated confidence bounds.
LightGBM/XGBoost quantile bounds target 80% coverage but remain uncalibrated.
All omit uncertainty in supplied stage and fitted reference.

Sparse early-stage legendary buckets can fall back to 600, badly overpredicting
Cosmog or Meltan. Later-added evolutions change historical stage labels. Special
design groups broaden single-stage distribution. These motivate better references,
historical labels and explicit group reports; see research roadmap.
