from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from final_project.config import ExperimentConfig


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _to_loader(x, y, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict(model: nn.Module, x: np.ndarray, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
    return torch.argmax(logits, dim=1).cpu().numpy()


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    config: ExperimentConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(
        input_dim=x_train.shape[1],
        hidden_dims=config.mlp_hidden_dims,
        output_dim=num_classes,
        dropout=config.mlp_dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.mlp_learning_rate, weight_decay=config.mlp_weight_decay
    )

    train_loader = _to_loader(x_train, y_train, config.mlp_batch_size, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(config.mlp_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(x_val, dtype=torch.float32, device=device))
            val_loss = loss_fn(val_logits, torch.tensor(y_val, dtype=torch.long, device=device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.mlp_patience:
                break

    model.load_state_dict(best_state)
    return model, device, _predict

