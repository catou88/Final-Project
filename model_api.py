"""Model API shim that preserves the dashboard-facing interface.

This module exposes a small registry of model builders and a proxy
`train_and_evaluate_model` that currently delegates to the existing
implementation in `final_project.py`. The goal is to centralize model
factories and later move training logic here while keeping the
`dashboard.py` import surface stable.
"""
from typing import Dict

import final_project as project
from models.hist_gradient import build_model as build_hist_gradient


MODEL_REGISTRY: Dict[str, callable] = {
    "Histogram gradient boosting": lambda: build_hist_gradient(
        project.NUMERIC_FEATURES, project.CATEGORICAL_FEATURES
    )
}


def train_and_evaluate_model(modeling, max_rows=None, include_catboost=False):
    """Proxy to the project's existing `train_and_evaluate_model`.

    Keeps the same signature and return structure so `dashboard.py` can
    continue importing `final_project` as before. Later this function can
    be updated to call builders from `MODEL_REGISTRY` directly.
    """
    return project.train_and_evaluate_model(
        modeling, max_rows=max_rows, include_catboost=include_catboost
    )


def list_models():
    return list(MODEL_REGISTRY.keys())
