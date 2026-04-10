from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path
    target_column: str
    output_dir: Path = Path("artifacts")
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2

    ml_models: tuple[str, ...] = ()

    mlp_hidden_dims: tuple[int, ...] = (128, 64)
    mlp_dropout: float = 0.2
    mlp_learning_rate: float = 1e-3
    mlp_weight_decay: float = 1e-4
    mlp_batch_size: int = 64
    mlp_epochs: int = 80
    mlp_patience: int = 10

    metrics: tuple[str, ...] = field(
        default_factory=lambda: ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    )
