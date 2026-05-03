from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(x_train, y_train, random_seed: int, class_weight=None):
    model = LogisticRegression(
        max_iter=3000,
        random_state=random_seed,
        class_weight=class_weight,
    )
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, random_seed: int, class_weight=None):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=random_seed,
        class_weight=class_weight,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    return model


def balanced_sample_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    mapping = {cls: weight for cls, weight in zip(classes, class_weights)}
    return np.array([mapping[y] for y in y_train], dtype=float)


def train_xgboost(x_train, y_train, random_seed: int, sample_weight=None):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the XGBoost baseline. Install it with 'pip install xgboost'."
        ) from exc

    num_classes = int(len(np.unique(y_train)))
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_seed,
        n_jobs=1,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model
