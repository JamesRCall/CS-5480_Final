from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(x_train, y_train, random_seed: int):
    model = LogisticRegression(
        max_iter=3000,
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, random_seed: int):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=random_seed,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    return model
