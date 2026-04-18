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

## In Progress

- Running experiments on the final Kaggle dataset file.
- Tuning MLP settings (layers, dropout, learning rate, epochs).
- Comparing deep model vs baseline models.

## Next

- Add a simple `predict.py` script for inference.
- Make one EDA notebook with key plots for the report.
- Add per-class results and better error analysis.
- Run with multiple random seeds for stable results.
- Prepare final report tables and charts.

## Notes

- Dataset is synthetic, so we should mention this in the report.
- Main target now is `burnout_level`.
