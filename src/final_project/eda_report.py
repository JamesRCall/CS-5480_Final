from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create EDA figures for the final report.")
    parser.add_argument("--data", required=True, help="Path to input CSV dataset.")
    parser.add_argument(
        "--target",
        default="burnout_level",
        help="Target column for class balance and breakdown charts.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/eda",
        help="Directory to save EDA figures and summary files.",
    )
    return parser.parse_args()


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, file_name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / file_name, bbox_inches="tight")
    plt.close(fig)


def _class_balance(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    counts = df[target].astype(str).value_counts().sort_index()
    perc = (counts / counts.sum() * 100.0).round(2)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=["#2C7FB8", "#41AB5D", "#F03B20"])
    ax.set_title(f"Target Class Balance ({target})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for bar, pct in zip(bars, perc.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    _save(fig, output_dir, "fig01_target_class_balance.png")

    pd.DataFrame({"class": counts.index, "count": counts.values, "percent": perc.values}).to_csv(
        output_dir / "target_class_balance.csv", index=False
    )


def _missingness(df: pd.DataFrame, output_dir: Path) -> None:
    missing_count = df.isna().sum().sort_values(ascending=False)
    missing_rate = (missing_count / len(df) * 100.0).round(2)
    missing = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percent": missing_rate.values,
        }
    )
    missing.to_csv(output_dir / "missingness_summary.csv", index=False)

    top_missing = missing.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_missing["column"][::-1], top_missing["missing_percent"][::-1], color="#7A5195")
    ax.set_title("Missingness by Feature (Top 15)")
    ax.set_xlabel("Missing (%)")
    ax.set_ylabel("Feature")
    _save(fig, output_dir, "fig02_missingness_top15.png")


def _numeric_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    preferred = [
        "burnout_score",
        "anxiety_score",
        "depression_score",
        "stress_level",
        "sleep_hours",
        "work_hours_per_week",
    ]
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    selected = [c for c in preferred if c in numeric_cols]
    if len(selected) < 6:
        fallback = [c for c in numeric_cols if c not in selected]
        selected.extend(fallback[: max(0, 6 - len(selected))])
    selected = selected[:6]

    if not selected:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, col in enumerate(selected):
        ax = axes[idx]
        values = df[col].dropna()
        ax.hist(values, bins=30, color="#00876C", alpha=0.9, edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Numeric Feature Distributions", fontsize=16)
    _save(fig, output_dir, "fig03_numeric_distributions_grid.png")


def _categorical_breakdowns(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    candidates = ["work_mode", "company_size", "job_role", "gender", "has_therapy"]
    available = [c for c in candidates if c in df.columns and c != target]
    if not available:
        return

    for fig_idx, col in enumerate(available[:3], start=4):
        table = pd.crosstab(df[col].astype(str), df[target].astype(str), normalize="index") * 100.0
        table = table.sort_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(table.index))
        colors = ["#2C7FB8", "#41AB5D", "#F03B20", "#FEB24C", "#756BB1"]
        for i, class_name in enumerate(table.columns):
            values = table[class_name].values
            ax.bar(table.index, values, bottom=bottom, label=class_name, color=colors[i % len(colors)])
            bottom += values

        ax.set_title(f"{col} Breakdown by {target} (Row %)")
        ax.set_xlabel(col)
        ax.set_ylabel("Percent within category")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title=target)
        _save(fig, output_dir, f"fig0{fig_idx}_categorical_{col}_by_target.png")


def _mental_health_corr(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    keywords = ["stress", "anxiety", "depression", "burnout", "mental"]
    mental_cols = [c for c in numeric_cols if any(k in c.lower() for k in keywords)]

    score_first = sorted(mental_cols, key=lambda c: ("score" not in c.lower(), c.lower()))
    selected = score_first[:8]
    if len(selected) < 2:
        return

    corr = df[selected].corr(numeric_only=True)
    corr.to_csv(output_dir / "mental_health_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Heatmap: Numeric Mental-Health Features")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    _save(fig, output_dir, "fig07_mental_health_correlation_heatmap.png")


def _write_caveats(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    counts = df[target].astype(str).value_counts(normalize=True).sort_index() * 100.0
    skew_line = ", ".join([f"{k}: {v:.1f}%" for k, v in counts.items()])

    caveats = [
        "# Data Caveats",
        "",
        "1. Synthetic data caveat",
        "This dataset is synthetic, so patterns may look cleaner than real employee data.",
        "Results may not transfer directly to real workplaces.",
        "",
        "2. Potential leakage proxies",
        "Some features may be very close to the target concept (for example stress, anxiety, depression, and burnout scores).",
        "These can boost model performance but may reduce real-world usefulness if such fields are unavailable at prediction time.",
        "",
        "3. Class skew check",
        f"Observed target mix for {target}: {skew_line}",
        "Use macro metrics and confusion matrices so small classes are not hidden by majority-class performance.",
    ]
    (output_dir / "data_caveats.md").write_text("\n".join(caveats), encoding="utf-8")


def run(data_path: Path, target: str, output_dir: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    output_dir.mkdir(parents=True, exist_ok=True)
    _set_style()

    _class_balance(df, target, output_dir)
    _missingness(df, output_dir)
    _numeric_distributions(df, output_dir)
    _categorical_breakdowns(df, target, output_dir)
    _mental_health_corr(df, output_dir)
    _write_caveats(df, target, output_dir)

    print(f"Saved EDA outputs to: {output_dir}")


def main() -> None:
    args = parse_args()
    run(data_path=Path(args.data), target=args.target, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
