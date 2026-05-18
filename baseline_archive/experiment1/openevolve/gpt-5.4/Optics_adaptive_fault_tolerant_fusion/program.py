# EVOLVE-BLOCK-START
import numpy as np


def fuse_and_compute_dm_commands(
    slopes_multi: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray | None = None,
    max_voltage: float = 0.50,
) -> np.ndarray:
    """
    Robust multi-WFS fusion.

    Improvements over a plain mean:
    - tolerate NaN/Inf values
    - suppress per-slope outliers with median/MAD winsorization
    - down-weight sensors that disagree strongly with the consensus
    - optionally use static sensor weights from control_model
    """
    slopes = np.asarray(slopes_multi, dtype=float)
    if slopes.ndim == 1:
        slopes = slopes[None, :]

    valid = np.isfinite(slopes)
    if not np.any(valid):
        if prev_commands is not None:
            return np.clip(np.asarray(prev_commands, dtype=float), -max_voltage, max_voltage)
        return np.zeros(reconstructor.shape[0], dtype=float)

    if slopes.shape[0] == 1:
        fused = np.where(valid[0], slopes[0], 0.0)
    else:
        masked = np.where(valid, slopes, np.nan)

        # Consensus estimate that is robust to a minority of bad sensors.
        center = np.nanmedian(masked, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)

        # Per-coordinate robust scale; clip only extreme deviations.
        residual = np.where(valid, slopes - center, np.nan)
        mad = np.nanmedian(np.abs(residual), axis=0)
        mad = np.where(np.isfinite(mad), mad, 0.0)
        clip_scale = 3.0 * (1.4826 * mad + 1e-6)

        clipped = np.where(
            valid,
            center + np.clip(slopes - center, -clip_scale, clip_scale),
            np.nan,
        )

        # Down-weight channels whose overall vector is inconsistent.
        sq_dev = np.where(valid, (clipped - center) ** 2, 0.0)
        counts = np.sum(valid, axis=1)
        row_valid = counts > 0
        row_dev = np.sqrt(np.sum(sq_dev, axis=1) / np.maximum(counts, 1))

        finite_row_dev = row_dev[row_valid]
        if finite_row_dev.size == 0:
            weights = row_valid.astype(float)
        else:
            row_scale = np.median(finite_row_dev)
            if not np.isfinite(row_scale) or row_scale <= 1e-12:
                weights = row_valid.astype(float)
            else:
                weights = 1.0 / (1.0 + (row_dev / (2.5 * row_scale + 1e-12)) ** 2)
                weights = np.where(row_valid & np.isfinite(weights), weights, 0.0)

        # Honor optional prior WFS reliability weights if present.
        if isinstance(control_model, dict):
            static_weights = control_model.get("sensor_weights", control_model.get("wfs_weights"))
            if static_weights is not None:
                sw = np.asarray(static_weights, dtype=float).reshape(-1)
                if sw.size == weights.size:
                    weights = weights * np.clip(sw, 0.0, None)

        weighted_clipped = np.where(valid, clipped, 0.0) * weights[:, None]
        weight_sum = np.sum(weights[:, None] * valid, axis=0)
        fused = np.where(
            weight_sum > 1e-12,
            np.sum(weighted_clipped, axis=0) / weight_sum,
            center,
        )

    u = reconstructor @ fused
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
