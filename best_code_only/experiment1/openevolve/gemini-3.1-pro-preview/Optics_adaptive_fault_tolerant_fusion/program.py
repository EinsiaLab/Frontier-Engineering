# EVOLVE-BLOCK-START
import numpy as np


def fuse_and_compute_dm_commands(
    slopes_multi: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray | None = None,
    max_voltage: float = 0.50,
) -> np.ndarray:
    d = np.sum(np.abs(slopes_multi[:, None] - slopes_multi), axis=2)
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(d.argmin(), d.shape)
    return np.clip(reconstructor @ (slopes_multi[i] + slopes_multi[j]) / 2, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
