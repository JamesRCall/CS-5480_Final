# CS-5480 Group 10 Final Project

Deep learning pipeline for predicting employee stress/burnout level from tabular data.

## What This Repo Does Now

- Loads a CSV dataset
- Preprocesses numeric + categorical features consistently
- Trains a feedforward PyTorch MLP for 3-class prediction
- Reports accuracy, macro precision/recall/F1, confusion matrix
- Saves model + preprocessing artifacts for later inference work

Optional: add baseline ML models with `--include-baselines`.

## Project Structure

- `src/final_project/` training and evaluation code
- `data/` dataset location
- `artifacts/` outputs from each run

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH='src'
python -m final_project.run_experiment --data data/employee_stress.csv --target burnout_level
```

With baselines:

```powershell
python -m final_project.run_experiment --data data/employee_stress.csv --target burnout_level --include-baselines
```

`--include-baselines` runs:

- Logistic Regression
- Logistic Regression (`class_weight='balanced'`)
- Random Forest
- Random Forest (`class_weight='balanced'`)
- XGBoost
- XGBoost (balanced sample weights)

Baseline comparison is saved to `artifacts/baseline_comparison.csv` and ranked by `f1_macro`.

## Run EDA (Generate Graphs)

Use this command to generate EDA figures for the report:

```powershell
$env:PYTHONPATH='src'
python -m final_project.eda_report --data data/employee_stress.csv --target burnout_level --output-dir artifacts/eda
```

This creates report-ready graphs and EDA summary files in `artifacts/eda/`.

Main figures:

- `fig01_target_class_balance.png`
- `fig02_missingness_top15.png`
- `fig03_numeric_distributions_grid.png`
- `fig04_categorical_work_mode_by_target.png`
- `fig05_categorical_company_size_by_target.png`
- `fig06_categorical_job_role_by_target.png`
- `fig07_mental_health_correlation_heatmap.png`

## Outputs

- `artifacts/metrics_summary.csv`
- `artifacts/metrics_details.json`
- `artifacts/baseline_comparison.csv` (when baselines are enabled)
- `artifacts/confusion_matrix_mlp_torch.csv`
- `artifacts/models/mlp_torch.pt`
- `artifacts/models/preprocessing.pkl`

## Next Steps

1. Run on the project CSV and verify class balance + target labels.
2. Tune MLP settings in `src/final_project/config.py` (layers, dropout, epochs, LR).
3. Add a small `predict.py` script that loads `mlp_torch.pt` + `preprocessing.pkl` for single-row or batch inference.
4. Add baseline comparison when needed for the final report section.
