"""Metrics used by the benchmark task evaluators."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def pareto_mask(points: np.ndarray, maximize: Iterable[bool]) -> np.ndarray:
    """Return the non-dominated mask for a set of points."""
    values = np.asarray(points, dtype=float)
    maximize_mask = np.asarray(list(maximize), dtype=bool)
    if values.size == 0:
        return np.zeros(0, dtype=bool)

    # Convert to minimization so the dominance test is unified.
    minimization_values = values.copy()
    minimization_values[:, maximize_mask] *= -1.0

    keep = np.ones(len(minimization_values), dtype=bool)
    for i, candidate in enumerate(minimization_values):
        if not keep[i]:
            continue
        dominates = np.all(minimization_values <= candidate, axis=1) & np.any(
            minimization_values < candidate, axis=1
        )
        dominates[i] = False
        if np.any(dominates):
            keep[i] = False
            continue
        dominated = np.all(candidate <= minimization_values, axis=1) & np.any(
            candidate < minimization_values, axis=1
        )
        dominated[i] = False
        keep[dominated] = False
    return keep


def hypervolume_2d_min(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute the dominated hypervolume for a 2D minimization problem."""
    values = np.asarray(points, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if values.size == 0:
        return 0.0
    if values.shape[1] != 2:
        raise ValueError("Only 2D hypervolume is implemented.")

    feasible = np.all(values <= ref, axis=1)
    values = values[feasible]
    if len(values) == 0:
        return 0.0

    mask = pareto_mask(values, maximize=[False, False])
    frontier = values[mask]
    order = np.argsort(frontier[:, 0])
    frontier = frontier[order]

    hv = 0.0
    current_y = float(ref[1])
    ref_x = float(ref[0])
    for x_val, y_val in frontier:
        if y_val >= current_y:
            continue
        hv += max(0.0, ref_x - float(x_val)) * (current_y - float(y_val))
        current_y = float(y_val)
    return float(hv)
