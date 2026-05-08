#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from final_project.data import to_dense
from final_project.deep_model import MLPClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D visualizations for the trained MLP model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/employee_stress.csv"),
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory containing preprocessing.pkl and mlp_torch.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Directory to save 3D figures.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target column override. Defaults to training target from artifacts.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of rows to plot in 3D PCA view.",
    )
    parser.add_argument(
        "--max-layer-nodes",
        type=int,
        default=20,
        help="Maximum displayed nodes per layer in topology figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for row sampling.",
    )
    return parser.parse_args()


def _load_preprocessing_artifacts(model_dir: Path) -> dict:
    with (model_dir / "preprocessing.pkl").open("rb") as f:
        return pickle.load(f)


def _load_mlp_model(model_dir: Path, device: str) -> tuple[MLPClassifier, dict]:
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


def _predict_encoded(model: MLPClassifier, x_dense: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x_tensor = torch.tensor(x_dense, dtype=torch.float32, device=device)
        logits = model(x_tensor)
    return torch.argmax(logits, dim=1).cpu().numpy()


def _make_3d_projection_plot(
    coords: np.ndarray,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(16, 7))
    try:
        cmap = plt.colormaps.get_cmap("tab10").resampled(len(class_names))
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10", len(class_names))

    if y_true is not None:
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        for cls_idx, class_name in enumerate(class_names):
            idx = y_true == cls_idx
            if np.any(idx):
                ax1.scatter(
                    coords[idx, 0],
                    coords[idx, 1],
                    coords[idx, 2],
                    s=18,
                    alpha=0.8,
                    color=cmap(cls_idx),
                    label=f"True {class_name}",
                )
        ax1.set_title("3D PCA Projection by True Class")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_zlabel("PC3")
        ax1.legend(loc="upper left", fontsize=8)

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        for cls_idx, class_name in enumerate(class_names):
            idx = y_pred == cls_idx
            if np.any(idx):
                ax2.scatter(
                    coords[idx, 0],
                    coords[idx, 1],
                    coords[idx, 2],
                    s=18,
                    alpha=0.8,
                    color=cmap(cls_idx),
                    label=f"Pred {class_name}",
                )
        ax2.set_title("3D PCA Projection by Predicted Class")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("PC3")
        ax2.legend(loc="upper left", fontsize=8)

        fig.tight_layout()
        fig.savefig(output_dir / "mlp_pca_3d_true_vs_pred.png", bbox_inches="tight")
        plt.close(fig)

        fig_err = plt.figure(figsize=(8.5, 7))
        ax_err = fig_err.add_subplot(1, 1, 1, projection="3d")
        correct = y_true == y_pred
        ax_err.scatter(
            coords[correct, 0],
            coords[correct, 1],
            coords[correct, 2],
            s=18,
            alpha=0.75,
            color="#2ca02c",
            label="Correct",
        )
        ax_err.scatter(
            coords[~correct, 0],
            coords[~correct, 1],
            coords[~correct, 2],
            s=24,
            alpha=0.9,
            color="#d62728",
            label="Misclassified",
        )
        ax_err.set_title("3D PCA Projection: Prediction Correctness")
        ax_err.set_xlabel("PC1")
        ax_err.set_ylabel("PC2")
        ax_err.set_zlabel("PC3")
        ax_err.legend(loc="upper left")
        fig_err.tight_layout()
        fig_err.savefig(output_dir / "mlp_pca_3d_correctness.png", bbox_inches="tight")
        plt.close(fig_err)
        return

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for cls_idx, class_name in enumerate(class_names):
        idx = y_pred == cls_idx
        if np.any(idx):
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                coords[idx, 2],
                s=18,
                alpha=0.8,
                color=cmap(cls_idx),
                label=f"Pred {class_name}",
            )
    ax.set_title("3D PCA Projection by Predicted Class")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "mlp_pca_3d_pred_only.png", bbox_inches="tight")
    plt.close(fig)


def _make_3d_topology_plot(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    max_layer_nodes: int,
    output_dir: Path,
) -> None:
    layer_sizes = [input_dim, *hidden_dims, output_dim]
    display_sizes = [min(max_layer_nodes, size) for size in layer_sizes]
    layer_labels = ["Input"] + [f"Hidden {i+1}" for i in range(len(hidden_dims))] + ["Output"]

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111, projection="3d")

    positions: list[list[tuple[float, float, float]]] = []
    for layer_idx, node_count in enumerate(display_sizes):
        if node_count == 1:
            y_coords = np.array([0.0])
            z_coords = np.array([0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, node_count, endpoint=False)
            radius = 1.0 + 0.15 * min(layer_idx, 3)
            y_coords = radius * np.cos(angles)
            z_coords = radius * np.sin(angles)
        x_coords = np.full(node_count, layer_idx, dtype=float)
        color = "#1f77b4" if layer_idx not in (0, len(display_sizes) - 1) else "#ff7f0e"
        ax.scatter(x_coords, y_coords, z_coords, s=40, alpha=0.95, color=color)
        layer_pos = [(x_coords[i], y_coords[i], z_coords[i]) for i in range(node_count)]
        positions.append(layer_pos)
        shown = display_sizes[layer_idx]
        total = layer_sizes[layer_idx]
        ax.text(
            layer_idx,
            0.0,
            2.2,
            f"{layer_labels[layer_idx]}\n{shown}/{total} nodes",
            ha="center",
            va="center",
            fontsize=9,
        )

    for layer_idx in range(len(display_sizes) - 1):
        for p1 in positions[layer_idx]:
            for p2 in positions[layer_idx + 1]:
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color="#7f7f7f",
                    alpha=0.08,
                    linewidth=0.7,
                )

    ax.set_title("3D MLP Topology (Displayed Nodes Capped Per Layer)")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Angled side-view variant.
    ax.view_init(elev=16, azim=-74)
    ax.set_box_aspect((4.2, 1.0, 0.85))
    ax.set_position([0.02, 0.05, 0.96, 0.9])
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)
    fig.savefig(output_dir / "mlp_network_topology_3d.png", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    artifacts = _load_preprocessing_artifacts(args.model_dir)
    preprocessor = artifacts["preprocessor"]
    label_encoder = artifacts["label_encoder"]
    default_target = artifacts["target_column"]
    class_names = [str(x) for x in artifacts["class_names"]]
    target = args.target or default_target

    df = pd.read_csv(args.data)
    if target in df.columns:
        y_true_labels = df[target].astype(str).to_numpy()
        x_df = df.drop(columns=[target])
    else:
        y_true_labels = None
        x_df = df

    if len(x_df) == 0:
        raise ValueError("Dataset has no rows to visualize.")

    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    if len(x_df) > args.max_samples:
        sampled = x_df.sample(n=args.max_samples, random_state=args.seed)
        if y_true_labels is not None:
            y_series = pd.Series(y_true_labels, index=x_df.index)
            y_true_labels = y_series.loc[sampled.index].to_numpy()
        x_df = sampled

    x_t = preprocessor.transform(x_df)
    x_dense = to_dense(x_t)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = _load_mlp_model(args.model_dir, device)
    y_pred_enc = _predict_encoded(model, x_dense, device)

    if y_true_labels is not None:
        y_true_enc = label_encoder.transform(y_true_labels)
    else:
        y_true_enc = None

    pca = PCA(n_components=3, random_state=args.seed)
    coords = pca.fit_transform(x_dense)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _make_3d_projection_plot(coords, y_true_enc, y_pred_enc, class_names, args.output_dir)
    _make_3d_topology_plot(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        output_dim=int(checkpoint["num_classes"]),
        max_layer_nodes=args.max_layer_nodes,
        output_dir=args.output_dir,
    )
    print(f"Saved 3D model visualization figures to: {args.output_dir}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
