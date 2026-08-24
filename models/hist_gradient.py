"""Histogram-gradient model builder.

Provides a `build_model(numeric_features, categorical_features)` factory
that returns an sklearn-compatible Pipeline mirroring the project's
`HistGradientBoostingRegressor` pipeline.
"""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from ._shared import build_preprocessing


def build_model(numeric_features, categorical_features):
    preprocessing = build_preprocessing(numeric_features, categorical_features)
    boosted_trees = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.10,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", boosted_trees)])
