from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from final_project.config import ExperimentConfig


def load_dataframe(config: ExperimentConfig) -> pd.DataFrame:
    if not config.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{config.data_path}'. Put your CSV in data/ and rerun."
        )
    df = pd.read_csv(config.data_path)
    if config.target_column not in df.columns:
        raise ValueError(
            f"Target column '{config.target_column}' is missing. Available columns: {list(df.columns)}"
        )
    return df


def split_dataframe(
    df: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df.drop(columns=[config.target_column])
    y = df[config.target_column]

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y.astype(str),
    )

    val_ratio_of_trainval = config.val_size / (1.0 - config.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_ratio_of_trainval,
        random_state=config.random_seed,
        stratify=y_trainval.astype(str),
    )

    train_df = x_train.copy()
    train_df[config.target_column] = y_train
    val_df = x_val.copy()
    val_df[config.target_column] = y_val
    test_df = x_test.copy()
    test_df[config.target_column] = y_test

    return train_df, val_df, test_df


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def prepare_xy(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[target_column])
    y = df[target_column].astype(str)
    return x, y


def encode_labels(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    y_test_enc = encoder.transform(y_test)
    return encoder, y_train_enc, y_val_enc, y_test_enc


def transform_features(
    preprocessor: ColumnTransformer, x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame
):
    x_train_t = preprocessor.fit_transform(x_train)
    x_val_t = preprocessor.transform(x_val)
    x_test_t = preprocessor.transform(x_test)
    return x_train_t, x_val_t, x_test_t


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if transformer == "passthrough":
            feature_names.extend(list(columns))
            continue
        if hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
            encoder = transformer.named_steps["encoder"]
            feature_names.extend(encoder.get_feature_names_out(columns).tolist())
        else:
            feature_names.extend(list(columns))
    return feature_names


def to_dense(array_like):
    return array_like.toarray() if hasattr(array_like, "toarray") else array_like
