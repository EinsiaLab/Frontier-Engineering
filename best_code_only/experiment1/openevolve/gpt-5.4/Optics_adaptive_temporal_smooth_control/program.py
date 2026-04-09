# EVOLVE-BLOCK-START
import numpy as np


def compute_dm_commands(
    slopes: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray,
    max_voltage: float = 0.25,
) -> np.ndarray:
    u = reconstructor @ slopes
    g = float(control_model.get("gain", 0.7))
    a = max(0.0, float(control_model.get("adaptive_alpha", 0.15)))
    sr = float(control_model.get("slew_rate", 0.02))
    d = float(control_model.get("damping", 0.0))
    g /= 1.0 + a * float(np.mean(slopes * slopes))
    du = np.clip(g * (u - prev_commands) - d * prev_commands, -sr, sr)
    return np.clip(prev_commands + du, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
