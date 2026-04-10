# EVOLVE-BLOCK-START
import numpy as np


def fuse_and_compute_dm_commands(
    slopes_multi: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray | None = None,
    max_voltage: float = 0.50,
) -> np.ndarray:
    x = np.asarray(slopes_multi, float)
    if x.ndim == 1 or x.shape[0] < 2:
        fused = x.reshape(-1) if x.ndim == 1 else x[0]
    else:
        c = np.median(x, 0)
        r = np.mean((x - c) ** 2, 1) + 0.3 * np.mean(x == 0, 1)
        m = control_model.get("anomaly_model") if isinstance(control_model, dict) else None
        if m is not None:
            s = m.decision_function(x)
            r = r - 0.2 * (s - s.mean()) / (s.std() + 1e-12)
        k = max(1, int(np.ceil((control_model.get("inlier_fraction", 0.4) if isinstance(control_model, dict) else 0.4) * x.shape[0])))
        fused = np.median(x[np.argpartition(r, k - 1)[:k]], 0)
    return np.clip(reconstructor @ fused, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
