"""Utility helpers shared by baselines, references, and evaluators."""

from __future__ import annotations

import json
import math
import random
from typing import Any

import numpy as np
import pandas as pd

from shared.summit_compat import apply_summit_compat


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy RNGs."""
    random.seed(seed)
    np.random.seed(seed)


def split_budget(total_budget: int, parts: int) -> list[int]:
    """Split an integer budget into near-equal positive chunks."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    base = total_budget // parts
    remainder = total_budget % parts
    chunks = [base + (1 if i < remainder else 0) for i in range(parts)]
    return [chunk for chunk in chunks if chunk > 0]


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value into a closed interval."""
    return float(max(lower, min(upper, value)))


def to_python(value: Any) -> Any:
    """Convert NumPy / pandas scalar types into plain Python values."""
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return None
        return to_python(value.iloc[0])
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def evaluate_candidate(experiment, candidate: dict[str, Any]) -> dict[str, Any]:
    """Run a single candidate through a Summit experiment and return a plain record."""
    apply_summit_compat()
    from summit.utils.dataset import DataSet

    frame = pd.DataFrame([candidate])
    result = experiment.run_experiments(DataSet.from_df(frame))
    row = result.iloc[-1]
    record: dict[str, Any] = {}
    for column in result.columns.get_level_values(0).unique():
        key = (column, "DATA") if (column, "DATA") in result.columns else column
        try:
            value = row[key]
        except Exception:
            continue
        record[column] = to_python(value)
    return record


def records_from_experiment_data(experiment, input_names: list[str], objective_names: list[str]) -> list[dict[str, Any]]:
    """Convert Summit experiment history into plain Python records."""
    data = experiment.data.reset_index(drop=True)
    records: list[dict[str, Any]] = []
    keep = input_names + objective_names
    for _, row in data.iterrows():
        record = {}
        for name in keep:
            key = (name, "DATA") if (name, "DATA") in data.columns else name
            record[name] = to_python(row[key])
        records.append(record)
    return records


def score_summary(values: list[float]) -> dict[str, float]:
    """Return mean, std, min, and max for a list of scores."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def dump_json(data: dict[str, Any]) -> str:
    """Serialize task results with stable formatting."""
    return json.dumps(
        data,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=to_python,
    )


def gaussian_perturbation(rng: np.random.Generator, scale: float) -> float:
    """Sample a scalar Gaussian perturbation."""
    return float(rng.normal(loc=0.0, scale=scale))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a finite ratio when denominator may be zero."""
    if math.isclose(denominator, 0.0):
        return 0.0
    return float(numerator / denominator)
