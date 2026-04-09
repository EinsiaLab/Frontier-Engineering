# EVOLVE-BLOCK-START
import numpy as np


def compute_dm_commands(
    slopes: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray,
    max_voltage: float = 0.25,
) -> np.ndarray:
    # Direct reconstructor with tuned exponential smoothing
    u = reconstructor @ slopes
    u = prev_commands + 0.8 * (u - prev_commands)
    
    # Explicitly bound the command slew to match the physical plant limit (0.055)
    return np.clip(np.clip(u, prev_commands - 0.055, prev_commands + 0.055), -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
