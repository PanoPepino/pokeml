# How training and checking work

A fold is one train/check round. Each round starts with a fresh model.

| Round | Train generations | Check generation | All check rows | Ordinary check rows |
|---|---|---|---:|---:|
| 1 | 1–5 | 6 | 72 | 72 |
| 2 | 1–6 | 7 | 106 | 95 |
| 3 | 1–7 | 8 | 133 | 133 |
| 4 | 1–8 | 9 | 122 | 102 |

Ordinary means everyone except 11 Ultra Beasts and 20 Paradox Pokémon. It does
not mean regular rarity: ordinary legendary Pokémon remain included. Poipole
and Naganadel belong to Ultra Beast group despite having evolution stages.
Group membership comes from fixed species identity, not BST or prediction error.
It is evaluation metadata, not model input. Stage and rarity stay unchanged.

Training always includes earlier-generation special groups. Ultra Beasts enter
rounds 3 and 4. Each round learns feature rules and median references only from
training data. Direct models learn BST. Correction models learn actual BST minus
training reference and add reference back when predicting.

New workflow runs use `selection_scope="ordinary"` by default. Hyperparameter
search and model selection use the same group and counts. Use `"all"` to compare
previous policy. Models predict all validation rows after choosing settings;
reports show all, ordinary, Ultra Beast and Paradox scores. No scored validation
labels select an early stopping round. Empty groups have count zero and no score;
primary group must contain rows in each of four tuning folds.

MAE is average absolute error, measured in BST points. Combine group/fold MAEs
by counts: sum(count × MAE) / sum(count). RMSE gives larger errors more weight:
sqrt(sum(count × RMSE²) / sum(count)). Positive average signed error (actual minus
prediction) means model predicts too low. Never apply square root to MAE.

Final selected model is fitted on generations 1–9, including special groups.
Its own training predictions are not validation scores. All four check generations
help select settings/model; no untouched final test exists. Results use current
stats/chains and are retrospective, not proof of future prediction quality.

Historical notebook mode reads old predictions and writes new group summaries
under `artifacts/evaluation/special_groups/`. It never changes old winner, scores,
model files or release. Historical filtered scores are a new breakdown, not
new training or an independent test.

Python: `WorkflowConfig(selection_scope="ordinary")`.
CLI: `python -m pokeml.cli model workflow --selection-scope ordinary --run-name my_unique_run`.
Full-population comparison: replace `ordinary` with `all`.
