from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def save_confusion_matrix(
    y_true, y_pred, class_names: list[str], output_dir: Path, model_name: str
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(output_dir / f"confusion_matrix_{model_name}.csv", index=True)


def save_metrics(results: list[dict], output_dir: Path, primary_metric: str = "f1_macro") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    baseline_df = metrics_df[metrics_df["model"] != "mlp_torch"].copy()
    if not baseline_df.empty and primary_metric in baseline_df.columns:
        baseline_df = baseline_df.sort_values(primary_metric, ascending=False).reset_index(drop=True)
        baseline_df.insert(0, "rank_by_f1_macro", baseline_df.index + 1)
        baseline_df["primary_metric"] = primary_metric
        baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)

    (output_dir / "metrics_details.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

