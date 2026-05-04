#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from final_project.data import to_dense
from final_project.deep_model import MLPClassifier


def load_preprocessing_artifacts(model_dir: Path) -> dict:
    """Load preprocessing artifacts from disk."""
    with (model_dir / "preprocessing.pkl").open("rb") as f:
        artifacts = pickle.load(f)
    return artifacts


def load_mlp_model(model_dir: Path, device: str = "cpu") -> tuple[MLPClassifier, dict]:
    """Load the trained MLP PyTorch model."""
    checkpoint = torch.load(model_dir / "mlp_torch.pt", map_location=device, weights_only=True)

    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        output_dim=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def load_baseline_model(model_dir: Path, model_name: str):
    with (model_dir / "{}.pkl".format(model_name)).open("rb") as f:
        model = pickle.load(f)
    return model


def predict_with_mlp(model: MLPClassifier, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        outputs = model(x_tensor)
        _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()


def validate_input_data(data_frame: pd.DataFrame, preprocessor) -> None:
    if not hasattr(preprocessor, "feature_names_in_"):
        return

    expected_columns = list(preprocessor.feature_names_in_)
    missing_columns = [col for col in expected_columns if col not in data_frame.columns]
    if missing_columns:
        raise ValueError(
            "Input data is missing columns required for prediction: {}".format(
                ", ".join(missing_columns)
            )
        )


def get_default_output_path(data_path: Path, model_name: str) -> Path:
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return predictions_dir / f"{data_path.stem}_{model_name}_predictions.csv"


def predict(new_data_path: Path, model_name: str = "mlp_torch", model_dir: Path = Path("artifacts/models")) -> pd.DataFrame:
    if not new_data_path.exists():
        raise FileNotFoundError(
            f"Input CSV file not found: {new_data_path}.\n"
            "Please provide an existing file path, e.g. data/employee_stress.csv"
        )

    artifacts = load_preprocessing_artifacts(model_dir)
    preprocessor = artifacts["preprocessor"]
    label_encoder = artifacts["label_encoder"]
    target_column = artifacts["target_column"]
    class_names = artifacts["class_names"]

    new_df = pd.read_csv(new_data_path)

    if target_column in new_df.columns:
        print("Warning: Target column '{}' found in data. Removing it for prediction.".format(target_column))
        new_df = new_df.drop(columns=[target_column])

    validate_input_data(new_df, preprocessor)

    x_new_transformed = preprocessor.transform(new_df)
    x_new_dense = to_dense(x_new_transformed)

    if model_name == "mlp_torch":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_mlp_model(model_dir, device)
        predictions_encoded = predict_with_mlp(model, x_new_dense, device)

    elif model_name in ["logistic_regression", "random_forest", "xgboost"]:
        try:
            model = load_baseline_model(model_dir, model_name)
            predictions_encoded = model.predict(x_new_dense)
        except FileNotFoundError:
            raise ValueError("Baseline model '{}' not found. Train it first using run_experiment.py".format(model_name))

    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    predictions = label_encoder.inverse_transform(predictions_encoded)

    result_df = new_df.copy()
    result_df["predicted_burnout_level"] = predictions

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions with trained burnout classification models"
    )
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=Path("data/employee_stress.csv"),
        help="Path to CSV file containing data for prediction (default: data/employee_stress.csv)",
    )
    parser.add_argument(
        "--model",
        choices=["mlp_torch", "logistic_regression", "random_forest", "xgboost"],
        default="mlp_torch",
        help="Model to use for prediction (default: mlp_torch)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory containing saved models (default: artifacts/models)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save predictions CSV (default: predictions/<file>_<model>_predictions.csv)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save predictions to the predictions/ folder using an auto-generated filename.",
    )

    args = parser.parse_args()
    predictions_df = predict(args.data_path, args.model, args.model_dir)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    elif args.save:
        output_path = get_default_output_path(args.data_path, args.model)
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        print("Predictions:")
        print(predictions_df.to_string())


if __name__ == "__main__":
    main()