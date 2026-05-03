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
- Added README with run instructions.
- Added `predict.py` inference support for `mlp_torch`, `logistic_regression`, and `random_forest`.
- Added default prediction saving to `predictions/`.
- Added per-class classification report export (`classification_report_<model>.csv`).
- Added baseline model saving so all saved models can be used for inference.
- Added multi-seed experiment runner with aggregated seed metrics.
- Added report generator for tables and charts.

## In Progress

- Running experiments on the final Kaggle dataset file.
- Tuning MLP settings (layers, dropout, learning rate, epochs).
- Comparing deep model vs baseline models.

## Next

- Generate final report figures and include them in the project writeup.
- Review model performance across multiple seeds and write analysis.
- Add EDA notebook or plots for the final report.

## Notes

- Dataset is synthetic, so we should mention this in the report.
- Main target now is `burnout_level`.
