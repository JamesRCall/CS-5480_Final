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


def save_metrics(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "metrics_details.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

