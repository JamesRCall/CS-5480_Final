from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from final_project.baselines import train_logistic_regression, train_random_forest
from final_project.config import ExperimentConfig
from final_project.data import (
    build_preprocessor,
    encode_labels,
    get_feature_names,
    load_dataframe,
    prepare_xy,
    split_dataframe,
    to_dense,
    transform_features,
)
from final_project.deep_model import train_mlp
from final_project.evaluate import classification_metrics, save_confusion_matrix, save_metrics


def run(config: ExperimentConfig):
    df = load_dataframe(config)
    train_df, val_df, test_df = split_dataframe(df, config)

    x_train_df, y_train = prepare_xy(train_df, config.target_column)
    x_val_df, y_val = prepare_xy(val_df, config.target_column)
    x_test_df, y_test = prepare_xy(test_df, config.target_column)

    preprocessor = build_preprocessor(x_train_df)
    x_train_t, x_val_t, x_test_t = transform_features(preprocessor, x_train_df, x_val_df, x_test_df)
    x_train = to_dense(x_train_t)
    x_val = to_dense(x_val_t)
    x_test = to_dense(x_test_t)

    label_encoder, y_train_enc, y_val_enc, y_test_enc = encode_labels(y_train, y_val, y_test)
    class_names = label_encoder.classes_.tolist()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    if "logistic_regression" in config.ml_models:
        logreg = train_logistic_regression(x_train, y_train_enc, config.random_seed)
        preds = logreg.predict(x_test)
        metrics = classification_metrics(y_test_enc, preds)
        results.append({"model": "logistic_regression", **metrics})
        save_confusion_matrix(y_test_enc, preds, class_names, output_dir, "logistic_regression")

    if "random_forest" in config.ml_models:
        rf = train_random_forest(x_train, y_train_enc, config.random_seed)
        preds = rf.predict(x_test)
        metrics = classification_metrics(y_test_enc, preds)
        results.append({"model": "random_forest", **metrics})
        save_confusion_matrix(y_test_enc, preds, class_names, output_dir, "random_forest")

    mlp_model, device, predict_fn = train_mlp(
        x_train=x_train,
        y_train=y_train_enc,
        x_val=x_val,
        y_val=y_val_enc,
        num_classes=len(class_names),
        config=config,
    )
    mlp_preds = predict_fn(mlp_model, x_test, device)
    mlp_metrics = classification_metrics(y_test_enc, mlp_preds)
    results.append({"model": "mlp_torch", **mlp_metrics})
    save_confusion_matrix(y_test_enc, mlp_preds, class_names, output_dir, "mlp_torch")

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Save core training artifacts for future inference/finetuning work.
    with (model_dir / "preprocessing.pkl").open("wb") as f:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "label_encoder": label_encoder,
                "target_column": config.target_column,
                "class_names": class_names,
            },
            f,
        )
    import torch

    torch.save(
        {
            "state_dict": mlp_model.state_dict(),
            "input_dim": x_train.shape[1],
            "hidden_dims": config.mlp_hidden_dims,
            "num_classes": len(class_names),
            "dropout": config.mlp_dropout,
            "target_column": config.target_column,
            "class_names": class_names,
        },
        model_dir / "mlp_torch.pt",
    )

    save_metrics(results, output_dir)

    feature_names = get_feature_names(preprocessor)
    (output_dir / "feature_count.txt").write_text(
        f"{len(feature_names)} transformed features\n", encoding="utf-8"
    )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ML vs DL comparison for employee stress/burnout prediction."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--target",
        default="burnout_level",
        help="Target column name (e.g., burnout_level or stress_level).",
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory for run outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also train logistic regression and random forest baselines.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        data_path=Path(args.data),
        target_column=args.target,
        output_dir=Path(args.output_dir),
        ml_models=("logistic_regression", "random_forest") if args.include_baselines else (),
        random_seed=args.seed,
    )
    results = run(config)
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
