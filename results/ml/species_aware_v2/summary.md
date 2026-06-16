# Species-Aware ML Baselines v2

- classification target: `observed_in_structure_label`
- regression target: `solvent_content_percent`
- grouped entries: `3939`
- species-aware rows: `11761`

## Headline classification result

- best combined model: `random_forest`
- grouped CV ROC-AUC: `0.956 +/- 0.002`
- grouped CV AP: `0.880 +/- 0.007`
- grouped CV balanced accuracy: `0.884 +/- 0.002`

## Headline regression result

- best combined model: `random_forest`
- grouped CV R2: `0.455 +/- 0.050`
- grouped CV MAE: `4.016 +/- 0.066`
- grouped CV RMSE: `6.709 +/- 0.235`

## Output files

- `classification_model_comparison.csv`
- `regression_model_comparison.csv`
- `classification_feature_importance.csv`
- `regression_feature_importance.csv`