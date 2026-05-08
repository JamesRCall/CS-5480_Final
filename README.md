# CS-5480 Group 10 Final Project

Deep learning pipeline for predicting employee stress/burnout level from tabular data.

## What This Repo Does Now

- Loads a CSV dataset
- Preprocesses numeric + categorical features consistently
- Trains a feedforward PyTorch MLP for 3-class prediction
- Compares performance against three baseline models: logistic regression, random forest, and XGBoost
- Reports accuracy, macro precision/recall/F1, confusion matrices, and per-class classification reports
- Saves model + preprocessing artifacts for later inference work

Use `--include-baselines` to include all three baselines during training.

## Project Structure

- `src/final_project/` training and evaluation code
- `data/` dataset location
- `artifacts/` outputs from each run

## Standard workflow

These commands are the simplest way to run the project.

### 1. Set up the environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the models

Run the full pipeline for all models, including the MLP and baseline classifiers: logistic regression, random forest, and XGBoost.

```powershell
python -m final_project.run_experiment --include-baselines --data data/employee_stress.csv --target burnout_level
```

### 3. Make predictions

Use the saved models to generate predictions for any model.

```powershell
python -m final_project.predict --save
```

To predict with a specific model:

```powershell
python -m final_project.predict --model random_forest --save
python -m final_project.predict --model xgboost --save
```

### 4. Run multiple random seeds

The runner uses three automatically selected random seeds by default. You can also pass your own seeds if desired.

```powershell
python -m final_project.run_multi_seed --include-baselines
```

### 5. Generate final reports and charts

```powershell
python -m final_project.generate_report
```

### 5b. Generate 3D model visualizations

```powershell
python -m final_project.model_3d_viz --data data/employee_stress.csv --model-dir artifacts/models --output-dir artifacts/reports
```

### 6. Generate EDA outputs

```powershell
python -m final_project.eda_report --data data/employee_stress.csv --target burnout_level
```

This creates report-ready graphs and EDA summary files in `artifacts/eda/`.

Main figures:

- `fig01_target_class_balance.png`
- `fig02_feature_type_distribution.png`
- `fig03_numeric_distributions_grid.png`
- `fig04_categorical_work_mode_by_target.png`
- `fig05_categorical_company_size_by_target.png`
- `fig06_categorical_job_role_by_target.png`
- `fig07_mental_health_correlation_heatmap.png`

## Outputs

- `artifacts/metrics_summary.csv`
- `artifacts/metrics_details.json`
- `artifacts/confusion_matrix_mlp_torch.csv`
- `artifacts/classification_report_mlp_torch.csv`
- `artifacts/models/mlp_torch.pt`
- `artifacts/models/preprocessing.pkl`
- `artifacts/eda/` — EDA charts and summaries
- `artifacts/reports/` — aggregate report tables and charts (`model_performance_summary.png`, `multi_seed_performance_summary.png`, `multi_seed_f1_summary.png`, `class_f1_comparison.png`, `mlp_actual_vs_predicted_counts.png`, `confusion_matrix_<model>_heatmap.png`)
- `artifacts/reports/` — includes 3D model visuals (`mlp_pca_3d_true_vs_pred.png`, `mlp_pca_3d_correctness.png`, `mlp_network_topology_3d.png`)
- `predictions/` — saved prediction CSVs

## File summary

- `src/final_project/run_experiment.py` — main training pipeline; trains the MLP and optionally baselines, saves artifacts, metrics, confusion matrices, and classification reports.
- `src/final_project/predict.py` — inference script that loads saved artifacts and predicts on new CSV input for `mlp_torch`, `logistic_regression`, `random_forest`, or `xgboost`.
- `src/final_project/run_multi_seed.py` — runs the experiment over multiple seeds and aggregates results, including XGBoost when `--include-baselines` is enabled.
- `src/final_project/generate_report.py` — builds summary tables and plots from saved metrics.
- `src/final_project/model_3d_viz.py` — builds 3D PCA and 3D neural-network topology visualizations for the trained MLP.
- `src/final_project/eda_report.py` — generates exploratory data analysis figures and data summaries.
- `src/final_project/evaluate.py` — metric calculation and export helper functions.
- `src/final_project/data.py` — data loading, splitting, preprocessing, and feature handling.
- `src/final_project/deep_model.py` — PyTorch MLP model definition, training loop, and inference helper.
- `src/final_project/baselines.py` — baseline training functions for logistic regression and random forest.
- `src/final_project/config.py` — experiment hyperparameters and configuration defaults.
- `src/final_project/train.py` — lightweight entrypoint that runs the main experiment.

## Quick commands

- `python -m final_project.run_experiment`
- `python -m final_project.run_experiment --include-baselines`
- `python -m final_project.predict --save`
- `python -m final_project.run_multi_seed --include-baselines`
- `python -m final_project.generate_report`
- `python -m final_project.model_3d_viz --data data/employee_stress.csv --model-dir artifacts/models --output-dir artifacts/reports`
- `python -m final_project.eda_report --data data/employee_stress.csv`
