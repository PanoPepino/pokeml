# EDA findings and candidate baseline decisions

All results use corrected minimal snapshot. EDA is retrospective; generation labels describe introduction while chains/stats describe current snapshot. No hard BST caps or forced type penalties are introduced.

## Stage buckets

| stage   |   count |   median |   std |
|:--------|--------:|---------:|------:|
| s1c2    |     237 |      308 | 54.19 |
| s1c3    |     116 |      300 | 40.36 |
| s2c2    |     264 |      487 | 38.44 |
| s2c3    |     118 |      405 | 59.67 |
| s3c3    |     125 |      523 | 42.94 |
| single  |     117 |      478 | 70.2  |

Stage separates much of BST variation. Single-stage bucket remains heterogeneous. Preserve starter status as candidate grouping, then validate correction features against baseline. Upper s3c3 outliers are eleven selected-form records at 600 plus Slaking at 670; cutoff 585. Statistical outlier does not mean invalid record.

## Generation drift

Mean per-fold individual-Pokémon MAE, equal weight across held-out generations 6–9. Primary population: 1,025 default species. Each forecast fits only earlier generations.

| stage   |   frozen_median |   linear_drift |   original_polynomial |   recent_3_median |
|:--------|----------------:|---------------:|----------------------:|------------------:|
| s3c3    |           25.53 |          25.9  |                 39.03 |             24.38 |
| single  |           47.41 |          48.32 |                 40.59 |             43.69 |

Cubic F3 hypothesis performs worse than frozen median; do not use it as default drift correction. Recent-three-generation median gives modest F3 gain. Original linear singles-from-Gen-3 proposal improves average error here, but gains vary by fold and this choice came from prior exploration. Keep it as candidate, not proven general rule.

|   generation |   actual_mean |   actual_median |   fixed_composition_mean |
|-------------:|--------------:|----------------:|-------------------------:|
|            1 |        400.92 |           405   |                   409.3  |
|            2 |        393.81 |           410   |                   400.84 |
|            3 |        386.74 |           405   |                   389.34 |
|            4 |        419.85 |           450   |                   407.38 |
|            5 |        409.22 |           418   |                   414.94 |
|            6 |        411.36 |           417   |                   416.34 |
|            7 |        429.22 |           469.5 |                   412.75 |
|            8 |        419.29 |           470   |                   407.26 |
|            9 |        435.9  |           479   |                   417.78 |

Fixed-composition means use same pooled stage weights in every generation. These are descriptive means, not weighted medians or forecasting features. Residual trends can still reflect type/archetype changes; composition adjustment does not establish causal power creep.

## Bug and type order

| stage   | bug   |   count |   families |   median_bst |   median_residual | sparse   |
|:--------|:------|--------:|-----------:|-------------:|------------------:|:---------|
| s1c2    | False |      71 |         71 |        320   |              12   | False    |
| s1c2    | True  |      19 |         19 |        304   |              -4   | False    |
| s1c3    | False |      38 |         38 |        291   |              -9   | False    |
| s1c3    | True  |       3 |          3 |        260   |             -40   | True     |
| s2c2    | False |     105 |        100 |        490   |               5   | False    |
| s2c2    | True  |      26 |         23 |        467.5 |             -17.5 | False    |
| s2c3    | False |      48 |         48 |        405   |               0   | False    |
| s2c3    | True  |       5 |          5 |        360   |             -45   | True     |
| s3c3    | False |      72 |         71 |        526.5 |               6.5 | False    |
| s3c3    | True  |       9 |          8 |        411   |            -109   | True     |
| single  | False |      64 |         64 |        487.5 |               9.5 | False    |
| single  | True  |       6 |          6 |        537.5 |              59.5 | True     |

Dual Bug medians are lower than other dual types in five stage groups. Single-stage group reverses direction: six Bug records have median 537.5 versus 487.5 for 64 other dual types. Several stage groups are below ten observations. Thus “worst at every stage” is not supported by this comparison. Partner/generation tables and family counts remain visible in typing notebook.

Exact swapped-pair comparisons: 38 stage/rarity/pair groups; 0 have at least ten observations in both directions. Slot-pool differences cannot establish ordering effects, and sparse exact comparisons do not justify a universal correction.

## Evidence register

| Hypothesis | Verdict | Modelling implication |
|---|---|---|
| Regular BST capped at 600 | Common pattern with Slaking exception; scope excludes many alternate forms | No hard cap |
| Stage establishes useful BST baseline | Supported descriptive separation | Fit medians within training cohorts |
| Shape explains stage residual | Exploratory association; type/family confounding remains | Test incremental forward gain |
| Rising overall median proves power creep | Composition and within-stage trends differ | Do not infer causal trend from overall median |
| Cubic F3 curve supports extrapolation | Challenged by forward errors | Reject default cubic correction |
| Singles linear drift since Gen 3 | Candidate improves mean fold MAE | Retain as research alternative |
| Bug pairing worst at every stage | Challenged by single-stage reversal and sparse groups | No forced Bug penalty |
| Primary/secondary order explains BST | Pooled partner compositions differ; exact groups sparse | Prefer raw types initially |

## Next modelling experiment

Compare stage/starter baseline, shared residual model and direct regression on identical forward cohorts. Treat community stage estimates separately from actual-stage oracle. Type-derived maps must be fold-local and cross-fitted. No EDA median, drift fit or whole-snapshot map becomes a training feature automatically.

Original exploratory hypotheses and relevant author notes remain in active notebook appendices. Superseded reports and migration records have been removed.

## Original exploratory hypotheses retained

Final three-stage means/medians were explored with cubic fits; single-stage trends
from generation 3 with linear fits. Notebook 02 reproduces and checks these against
simple chronological forecasts. High fitted R² across nine points is not proof of
extrapolation. Type-ordering deviations and stage-centred shape comparisons remain
EDA hypotheses; target-derived maps require fold-local fitting and cross-fitted
training encodings before prediction experiments.
