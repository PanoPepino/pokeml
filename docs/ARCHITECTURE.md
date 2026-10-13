# Package structure

| Module | Responsibility |
|---|---|
| data | Explicit acquisition, offline cleaning, pinned identity/evolution metadata, validation |
| eda | Stage, type, morphology and generation analysis |
| features | Registered feature fitting/application and model encodings |
| models | Median baseline, direct/correction regressors, four estimator variants, saved prediction |
| evaluation | Four chronological folds, weighted errors, special-group reporting, randomized search, training curves |
| pipeline | Input schema, shared transformations and workflow orchestration |
| visualisation / reporting | Plots and CLI tables |
| commands | Thin CLI adapters |

Training lives in `pipeline.workflow.run_workflow`. Notebook 04 and CLI call same
implementation. Features and baselines fit training rows only. IDs, generations
and target components are metadata, not predictor inputs. Registered starter tag
is enabled by default; target-derived type maps remain disabled in main workflow.

`PokemonPredictor.load(...).predict(rows)` loads saved fitted state without training.
Current tuned release uses direct CatBoost native. `predict_at_iterations()` exposes
partial-tree predictions for diagnostic curves. Both direct/correction candidates
and baseline-only reference use shared evaluation paths.

`load_workflow_result` requires matching current source and data. Explicit
`load_historical_workflow_result` reads frozen completed results after data checks;
it does not certify current-code performance. Notebook 04 defaults to saved tuned
results. Set `REUSE_RUN=None` to train. Notebook 05 reads pinned historical curves;
new curve generation requires explicit flag and unused directory.

New runs choose ordinary-Pokémon MAE; special designs remain in training and separate
reports. See [validation](VALIDATION.md), [methods](BASELINE_CORRECTION.md),
[data](DATA_PREPARATION.md), [findings](EDA_FINDINGS.md), and [research](RESEARCH_ROADMAP.md).

Eight active notebooks cover preparation, stage, drift, typing, training, curves,
shape and capture-rate research. Verification scripts execute them and verify
unchanged dataset bytes. Core preparation helpers remain for model transformations
and feature-validation tests. Installed environment currently uses project
`pokeml.egg-info`; metadata is refreshed after dependency changes.
