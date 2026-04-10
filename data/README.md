# Dataset Placement

Place the dataset CSV here, for example:

- `data/employee_stress.csv`

Required:

- A target column passed via `--target` (example: `burnout_level` or `stress_level`)
- Tabular feature columns (numeric and/or categorical)

Example:

```powershell
python -m final_project.run_experiment --data data/employee_stress.csv --target burnout_level
```
