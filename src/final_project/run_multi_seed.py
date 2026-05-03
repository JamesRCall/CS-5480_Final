#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from final_project.config import ExperimentConfig
from final_project.run_experiment import run


def run_multi_seed(
    data_path: Path,
    target_column: str,
    output_dir: Path,
    seeds: tuple[int, ...],
    include_baselines: bool,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        config = ExperimentConfig(
            data_path=data_path,
            target_column=target_column,
            output_dir=seed_dir,
            ml_models=("logistic_regression", "random_forest") if include_baselines else (),
            random_seed=seed,
        )
        print(f"Running seed={seed} -> output={seed_dir}")
        seed_results = run(config)
        for row in seed_results:
            row_with_seed = {**row, "seed": seed}
            all_results.append(row_with_seed)

    metrics_path = output_dir / "metrics_summary.csv"
    pd.DataFrame(all_results).to_csv(metrics_path, index=False)
    (output_dir / "metrics_details.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    aggregate = (
        pd.DataFrame(all_results)
        .groupby("model")
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
             precision_macro_mean=("precision_macro", "mean"), precision_macro_std=("precision_macro", "std"),
             recall_macro_mean=("recall_macro", "mean"), recall_macro_std=("recall_macro", "std"),
             f1_macro_mean=("f1_macro", "mean"), f1_macro_std=("f1_macro", "std"))
        .reset_index()
    )
    aggregate.to_csv(output_dir / "metrics_summary_aggregate.csv", index=False)
    return all_results


def resolve_data_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    fallback = Path("data") / raw_path
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Dataset not found at '{raw_path}' or '{fallback}'. Put your CSV in data/ and rerun."
    )


def resolve_random_seeds(seeds: list[int] | None, count: int = 3) -> tuple[int, ...]:
    if seeds:
        return tuple(seeds)
    generated = random.sample(range(1000, 9999), count)
    print(f"Generated random seeds: {generated}")
    return tuple(generated)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full experiment across multiple random seeds.")
    parser.add_argument(
        "--data",
        default="data/employee_stress.csv",
        help="Path to dataset CSV (default: data/employee_stress.csv).",
    )
    parser.add_argument("--target", default="burnout_level", help="Target column name.")
    parser.add_argument("--output-dir", default="artifacts/multi_seed", help="Directory for multi-seed outputs.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of random seeds to evaluate. If omitted, three random seeds are chosen.",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include baseline models in each seed run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_multi_seed(
        data_path=resolve_data_path(args.data),
        target_column=args.target,
        output_dir=Path(args.output_dir),
        seeds=resolve_random_seeds(args.seeds),
        include_baselines=args.include_baselines,
    )
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
