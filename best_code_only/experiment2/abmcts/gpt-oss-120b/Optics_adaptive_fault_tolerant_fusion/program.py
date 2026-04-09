# EVOLVE-BLOCK-START
import numpy as np


def _inverse_deviation_weights(norms: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute weights that give higher emphasis to sensors whose slope‑vector norm
    is close to the median. Sensors with larger deviation receive smaller
    weights via an inverse‑distance scheme.

    Parameters
    ----------
    norms : np.ndarray
        1‑D array of L2 norms for each sensor's slope vector.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalised weights that sum to 1.
    """
    median = np.median(norms)
    deviation = np.abs(norms - median)
    # Inverse‑distance weighting; add eps to keep finite values.
    inv = 1.0 / (deviation + eps)
    total = inv.sum()
    if total == 0 or np.isnan(total):
        # Degenerate case – fall back to uniform weighting.
        return np.full_like(inv, 1.0 / inv.size, dtype=float)
    return inv / total


def fuse_and_compute_dm_commands(
    slopes_multi: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray | None = None,
    max_voltage: float = 0.50,
) -> np.ndarray:
    """
    Robust fusion of multi‑sensor slope measurements and conversion to DM commands.

    The implementation follows a two‑step weighting scheme:
    1. Compute a quick robust weight based on the median absolute deviation of
       each sensor's slope‑vector norm.
    2. Refine the weights by measuring each sensor's deviation from the initial
       fused slope vector and applying an inverse‑distance weighting.

    An optional temporal smoothing blends the newly computed command vector
    with the previous command vector.

    Parameters
    ----------
    slopes_multi : np.ndarray
        Shape ``(n_sensors, n_slopes)`` containing slope measurements from each
        wave‑front sensor.
    reconstructor : np.ndarray
        Reconstruction matrix mapping fused slopes to DM commands.
    control_model : dict
        May contain tuning parameters:
        ``"smoothing"`` – temporal blending factor (0‑1),
        ``"weight_eps"`` – epsilon for inverse‑distance weighting.
    prev_commands : np.ndarray | None, optional
        Previously issued DM command vector for temporal smoothing.
    max_voltage : float, optional
        Maximum absolute voltage that can be applied to a DM actuator.

    Returns
    -------
    np.ndarray
        Voltage‑clipped DM command vector.
    """
    # Step 1: basic robustness weighting using median absolute deviation.
    sensor_norms = np.linalg.norm(slopes_multi, axis=1)
    basic_weights = _inverse_deviation_weights(sensor_norms)

    # Initial fused slopes using the basic weights.
    fused_initial = np.average(slopes_multi, axis=0, weights=basic_weights)

    # Step 2: refine weights by how far each sensor deviates from the initial fuse.
    residuals = np.linalg.norm(slopes_multi - fused_initial, axis=1)
    eps = control_model.get("weight_eps", 1e-6) if isinstance(control_model, dict) else 1e-6
    refined_weights = _inverse_deviation_weights(residuals, eps=eps)

    # Final fused slopes.
    fused = np.average(slopes_multi, axis=0, weights=refined_weights)

    # Linear reconstruction to DM commands.
    u_new = reconstructor @ fused

    # Temporal smoothing if a previous command vector is supplied.
    if prev_commands is not None:
        alpha = 0.5  # default blend factor
        if isinstance(control_model, dict) and "smoothing" in control_model:
            try:
                alpha = float(control_model["smoothing"])
                alpha = np.clip(alpha, 0.0, 1.0)
            except Exception:
                pass
        u = alpha * prev_commands + (1.0 - alpha) * u_new
    else:
        u = u_new

    # Clip to physical DM voltage limits.
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
