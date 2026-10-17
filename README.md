# Pokémon BST prediction

Can new Pokémon’s **BST**, i.e. the sum of its six base stats, be predicted using current Pokédex data?

Inputs: types, evolution stage, rarity, shape, colour, height, weight and starter status.
Stage and rarity are supplied assumptions. Actual BST and component stats never enter model inputs.

## Two methods

| Method | Prediction |
|---|---|
| Direct regression | Model predicts BST from attributes |
| Baseline + correction | Stage/rarity median + model’s predicted adjustment |

Each training round learns its own baseline. A median-only predictor is the simple reference to beat.

## Four models

Both methods compare **CatBoost native**, **CatBoost ordinal**, **LightGBM** and **XGBoost**.
Native CatBoost handles categories directly; ordinal variant uses numeric stage/rarity mappings.
These models combine decision trees through gradient boosting. Optional randomized search tunes their settings.

## Four training rounds

Each round trains fresh models on earlier generations and checks the next generation.

| Train generations | Check generation | Ordinary Pokémon checked |
|---|---|---:|
| 1–5 | 6 | 72 |
| 1–6 | 7 | 95 |
| 1–7 | 8 | 133 |
| 1–8 | 9 | 102 |

New runs choose settings/model using **ordinary-Pokémon MAE**: average absolute error in BST points,
pooled across 402 Pokémon. “Ordinary” excludes Ultra Beasts and Paradox Pokémon, regardless of rarity.
Reports also show all-Pokémon and separate special-group scores. Training retains every eligible row.

Baseline and feature rules learn from training data only. Final selected model is fitted on generations 1–9.
All four check generations guide selection: **no untouched final test exists**.
Current stats/evolution chains make results retrospective; family overlap and uncertain stage labels remain limitations.

## Predictions for three new Pokémon

<table>
  <tr>
    <td align="center"><img src="plots/browt.png" alt="Browt" width="260" style="border: 3px solid #78C850; border-radius: 12px;"><br><strong>Browt</strong></td>
    <td align="center"><img src="plots/pombon.png" alt="Pombon" width="260" style="border: 3px solid #F08030; border-radius: 12px;"><br><strong>Pombon</strong></td>
    <td align="center"><img src="plots/gecqua.png" alt="Gecqua" width="260" style="border: 3px solid #2DA1EF; border-radius: 12px;"><br><strong>Gecqua</strong></td>
  </tr>
</table>

<!-- PREDICTIONS -->

| Model | Ordinary validation MAE | Browt | Pombon | Gecqua |
|---|---:|---:|---:|---:|
| LightGBM — direct | 28.28 | 280 [246, 316] | 308 [311, 316] † | 310 [311, 312] † |
| CatBoost ordinal — direct | 28.42 | 276 ± 16 | 311 ± 12 | 312 ± 10 |
| CatBoost native — direct | 28.52 | 263 ± 24 | 305 ± 15 | 307 ± 13 |

<!-- /PREDICTIONS -->

Values are BST estimates. CatBoost uses **± one model-estimated standard deviation (sigma)**.
LightGBM uses **[lower, upper] nominal 80% prediction range**. These are different uncertainty
measures, are not calibrated guarantees, and are not measured prediction errors.
† LightGBM fits its bounds separately; these ranges do not contain its point estimate.

Models are ranked by ordinary-Pokémon errors from saved `pokemon_tuned_60_v1` experiments,
then refitted on generations 1–9 with frozen settings. No new tuning or validation was performed.
Original winning release remains unchanged. Predictions depend on supplied attributes and stage.

## Data pipeline

```text
Raw Pokémon data + saved reference data
                   |
                   v
          Clean and check data
                   |
                   v
           Small analysis table
                   |
          +--------+--------+
          |                 |
          v                 v
     Explore patterns   Four training rounds
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          Earlier generations     Next generation
                 |                     |
                 v                     |
        Build features and baseline    |
                 |                     |
                 v                     |
        Train models and try settings  |
                 |                     |
                 +----------+----------+
                            |
                            v
                 Predict next generation
                            |
                            v
                 Compare prediction errors
                 Ordinary + special groups
                            |
                            v
                Choose settings and model
                            |
                            v
                 Train on generations 1–9
                            |
                            v
                   Save fitted model
                            |
New Pokémon attributes -----+
                            |
                            v
                     Predicted BST
```

Feature rules and baseline learn from training rows only. All four check generations
help choose settings/model; none is an untouched final test.
