# Project Progress

## Done

- Set up project structure and Python package files.
- Added code to load CSV data.
- Added data split (train, validation, test).
- Added preprocessing for numeric and categorical columns.
- Added label encoding for target classes.
- Built and trained a PyTorch MLP model.
- Added early stopping in training.
- Added baseline models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Class-weight/balanced variants for baseline comparison
- Added evaluation metrics:
  - Accuracy
  - Macro precision
  - Macro recall
  - Macro F1
- Added confusion matrix export.
- Added artifact saving:
  - Model file (`mlp_torch.pt`)
  - Preprocessing file (`preprocessing.pkl`)
  - Metrics CSV and JSON files
- Added EDA report script (`src/final_project/eda_report.py`) and generated EDA figures.
- Added baseline comparison output file (`artifacts/baseline_comparison.csv`).
- Added README with run instructions.

## In Progress

- Running experiments on the final Kaggle dataset file.
- Tuning MLP settings (layers, dropout, learning rate, epochs).

## Next

- Add a `predict.py` script for inference.
- Add per-class results and better error analysis.
- Run with multiple random seeds for stable results.
- Prepare final report tables and charts.

## Baseline Comparison (Macro F1 Ranking)

| rank_by_f1_macro | model                        | accuracy           | precision_macro    | recall_macro       | f1_macro           | primary_metric |
| ---------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------------- |
| 1                | xgboost                      | 1.0                | 1.0                | 1.0                | 1.0                | f1_macro       |
| 2                | xgboost_balanced             | 1.0                | 1.0                | 1.0                | 1.0                | f1_macro       |
| 3                | logistic_regression_balanced | 0.9988             | 0.8807100859339666 | 0.9989957806019195 | 0.9274761212779549 | f1_macro       |
| 4                | random_forest_balanced       | 0.9998             | 0.9994548923412374 | 0.8461538461538461 | 0.8997272231314785 | f1_macro       |
| 5                | random_forest                | 0.9997             | 0.9991830065359477 | 0.7692307692307692 | 0.8231204138096957 | f1_macro       |
| 6                | logistic_regression          | 0.9996666666666667 | 0.9990924766312732 | 0.7435897435897436 | 0.791212286441294  | f1_macro       |

## Notes

- Dataset is synthetic, so we should mention this in the report.
- Main target now is `burnout_level`.
