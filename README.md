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

## Outputs

- `artifacts/metrics_summary.csv`
- `artifacts/metrics_details.json`
- `artifacts/confusion_matrix_mlp_torch.csv`
- `artifacts/models/mlp_torch.pt`
- `artifacts/models/preprocessing.pkl`

## Next Steps

1. Run on the project CSV and verify class balance + target labels.
2. Tune MLP settings in `src/final_project/config.py` (layers, dropout, epochs, LR).
3. Add a small `predict.py` script that loads `mlp_torch.pt` + `preprocessing.pkl` for single-row or batch inference.
4. Add baseline comparison when needed for the final report section.
