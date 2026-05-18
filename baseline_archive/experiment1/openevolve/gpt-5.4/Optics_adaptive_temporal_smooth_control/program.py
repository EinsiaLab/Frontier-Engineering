# EVOLVE-BLOCK-START
import numpy as np


def compute_dm_commands(
    slopes: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray,
    max_voltage: float = 0.25,
) -> np.ndarray:
    """Low-pass the reconstructed command and cap unreachable per-step jumps."""
    model = control_model or {}
    s = np.nan_to_num(np.asarray(slopes, dtype=float))
    prev = np.nan_to_num(np.asarray(prev_commands, dtype=float))
    target = np.nan_to_num(np.asarray(model.get("reconstructor", reconstructor) @ s, dtype=float))
    alpha = np.clip(np.asarray(model.get("gain", model.get("alpha", 0.35)), dtype=float), 0.0, 1.0)
    step = np.minimum(np.asarray(model.get("slew_rate", model.get("max_delta", 0.055)), dtype=float), 0.055)
    u = prev + np.clip(alpha * (target - prev), -step, step)
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
