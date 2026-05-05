#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(metrics_path: Path) -> pd.DataFrame | None:
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


def _get_plot_limits(
    values: pd.Series,
    min_margin: float = 0.05,
    max_margin: float = 0.05,
    cap_max: float | None = None,
) -> tuple[float, float]:
    min_value = max(0.0, values.min() - min_margin)
    max_value = values.max() + max_margin
    if cap_max is not None:
        max_value = min(max_value, cap_max)
    if max_value - min_value < 0.12:
        max_value = min_value + 0.12
        if cap_max is not None:
            max_value = min(max_value, cap_max)
    return min_value, max_value


def save_model_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_metrics_table.csv"
    metrics.to_csv(output_path, index=False)

    plot_data = metrics.set_index("model")[
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data.plot.bar(ax=ax, rot=0, width=0.8)
    y_min, y_max = _get_plot_limits(plot_data.stack(), min_margin=0.05, max_margin=0.08, cap_max=1.15)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Model performance on test set")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    label_map = {
        "logistic_regression": "LogReg",
        "logistic_regression_balanced": "LogReg B",
        "random_forest": "RF",
        "random_forest_balanced": "RF B",
        "xgboost": "XGB",
        "xgboost_balanced": "XGB B",
        "mlp_torch": "MLP",
    }
    ax.set_xticklabels([label_map.get(label, label) for label in plot_data.index], rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    for container in ax.containers:
        if hasattr(container, "patches"):
            ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    fig.savefig(output_dir / "model_performance_summary.png")
    plt.close()


def load_confusion_matrix(metrics_dir: Path, model_name: str) -> pd.DataFrame | None:
    path = metrics_dir / f"confusion_matrix_{model_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def load_classification_report(metrics_dir: Path, model_name: str) -> pd.DataFrame | None:
    path = metrics_dir / f"classification_report_{model_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def save_confusion_matrix_heatmaps(metrics_dir: Path, output_dir: Path, model_names: list[str]) -> None:
    label_map = {
        "logistic_regression": "LogReg",
        "logistic_regression_balanced": "LogReg B",
        "random_forest": "RF",
        "random_forest_balanced": "RF B",
        "xgboost": "XGB",
        "xgboost_balanced": "XGB B",
        "mlp_torch": "MLP",
    }

    for model_name in model_names:
        cm = load_confusion_matrix(metrics_dir, model_name)
        if cm is None or cm.empty:
            continue

        cm_values = cm.values.astype(float)
        row_totals = cm_values.sum(axis=1, keepdims=True)
        row_pct = np.divide(
            cm_values,
            row_totals,
            out=np.zeros_like(cm_values, dtype=float),
            where=row_totals > 0,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(row_pct, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns, rotation=30, ha="right")
        ax.set_yticklabels(cm.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        display_name = label_map.get(model_name, model_name)
        ax.set_title(f"Confusion Matrix (Row-Normalized): {display_name}")

        for i in range(cm_values.shape[0]):
            for j in range(cm_values.shape[1]):
                pct = row_pct[i, j] * 100.0
                count = int(cm_values[i, j])
                text_color = "white" if row_pct[i, j] >= 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{count}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row-normalized proportion")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix_{model_name}_heatmap.png", bbox_inches="tight")
        plt.close(fig)


def save_mlp_prediction_vs_actual(metrics_dir: Path, output_dir: Path) -> None:
    cm = load_confusion_matrix(metrics_dir, "mlp_torch")
    if cm is None:
        return

    actual_counts = cm.sum(axis=1)
    predicted_counts = cm.sum(axis=0)
    counts = pd.DataFrame(
        {
            "actual_count": actual_counts,
            "predicted_count": predicted_counts,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot.bar(ax=ax, rot=0)
    ax.set_title("MLP actual vs predicted class counts")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "mlp_actual_vs_predicted_counts.png")
    plt.close()


def save_class_f1_comparison(metrics_dir: Path, output_dir: Path, model_names: list[str]) -> None:
    records: list[dict[str, object]] = []
    for model_name in model_names:
        report = load_classification_report(metrics_dir, model_name)
        if report is None:
            print(f"Warning: No classification report found for {model_name}")
            continue

        report = report[~report.index.str.contains("macro", case=False, regex=True)]
        report = report[~report.index.str.contains("weighted", case=False, regex=True)]
        if report.empty:
            continue

        for class_label, row in report.iterrows():
            f1_value = None
            if "f1_score" in row:
                f1_value = row["f1_score"]
            elif "f1-score" in row:
                f1_value = row["f1-score"]
            else:
                continue

            records.append(
                {
                    "model": model_name,
                    "class": class_label,
                    "f1_score": float(f1_value),
                }
            )

    if not records:
        return

    data = pd.DataFrame(records)
    pivot = data.pivot(index="class", columns="model", values="f1_score")
    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot.bar(ax=ax, rot=0)
    y_min, y_max = _get_plot_limits(data["f1_score"], min_margin=0.08, max_margin=0.08, cap_max=1.03)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Per-class F1 comparison across models (single-run metrics)")
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Class")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Model")
    for container in ax.containers:
        if hasattr(container, "patches"):
            ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)
    plt.tight_layout()
    plt.savefig(output_dir / "class_f1_comparison.png")
    plt.close()


def save_multi_seed_report(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    agg = (
        metrics.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_macro_mean=("precision_macro", "mean"),
            precision_macro_std=("precision_macro", "std"),
            recall_macro_mean=("recall_macro", "mean"),
            recall_macro_std=("recall_macro", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
        )
        .reset_index()
    )
    agg.to_csv(output_dir / "multi_seed_metrics_aggregate.csv", index=False)

    metrics = [
        ("accuracy_mean", "accuracy_std", "Accuracy"),
        ("precision_macro_mean", "precision_macro_std", "Precision"),
        ("recall_macro_mean", "recall_macro_std", "Recall"),
        ("f1_macro_mean", "f1_macro_std", "F1 Score"),
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    x = range(len(agg))
    width = 0.18
    all_plot_values = []
    for idx, (mean_col, std_col, label) in enumerate(metrics):
        values = agg[mean_col]
        bar_positions = [pos + idx * width for pos in x]
        ax.bar(bar_positions, values, width, label=label)
        all_plot_values.extend(values.tolist())

    ax.set_title("Multi-seed average performance by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks([pos + 1.5 * width for pos in x])
    ax.set_xticklabels(agg["model"])
    y_min, y_max = _get_plot_limits(pd.Series(all_plot_values), min_margin=0.08, max_margin=0.04, cap_max=None)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    for container in ax.containers:
        if hasattr(container, "patches"):
            ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)

    plt.tight_layout()
    plt.savefig(output_dir / "multi_seed_performance_summary.png")
    plt.close()


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        agg["model"],
        agg["f1_macro_mean"],
        color="tab:orange",
    )
    y_min, y_max = _get_plot_limits(
        agg["f1_macro_mean"],
        min_margin=0.08,
        max_margin=0.04,
        cap_max=None,
    )
    ax.set_ylim(y_min, y_max)
    ax.set_title("Multi-seed average F1 score by model")
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for container in ax.containers:
        if hasattr(container, "patches"):
            ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2)
    plt.tight_layout()
    plt.savefig(output_dir / "multi_seed_f1_summary.png")
    plt.close()


def generate_report(
    metrics_dir: Path,
    multi_seed_dir: Path,
    output_dir: Path,
) -> None:
    base_metrics = load_metrics(metrics_dir / "metrics_summary.csv")
    multi_seed_metrics = load_metrics(multi_seed_dir / "metrics_summary.csv")

    if base_metrics is None and multi_seed_metrics is None:
        raise FileNotFoundError(
            f"No metrics summary found in {metrics_dir} or {multi_seed_dir}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if base_metrics is not None:
        save_model_summary(base_metrics, output_dir)
        model_names = base_metrics["model"].dropna().astype(str).unique().tolist()
        save_confusion_matrix_heatmaps(
            metrics_dir=metrics_dir,
            output_dir=output_dir,
            model_names=model_names,
        )
        save_class_f1_comparison(
            metrics_dir=metrics_dir,
            output_dir=output_dir,
            model_names=model_names,
        )
        save_mlp_prediction_vs_actual(metrics_dir, output_dir)
        print(f"Saved base model summary and additional charts to {output_dir}")

    if multi_seed_metrics is not None:
        save_multi_seed_report(multi_seed_metrics, output_dir)
        print(f"Saved multi-seed aggregate report and chart to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final report tables and charts.")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing the base metrics summary CSV.",
    )
    parser.add_argument(
        "--multi-seed-dir",
        type=Path,
        default=Path("artifacts/multi_seed"),
        help="Directory containing the multi-seed metrics summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Directory to save report outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_report(args.metrics_dir, args.multi_seed_dir, args.output_dir)


if __name__ == "__main__":
    main()
