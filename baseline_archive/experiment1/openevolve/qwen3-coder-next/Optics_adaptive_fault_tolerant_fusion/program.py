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
    Robust fusion: weighted average based on local variance estimates.
    
    Uses inverse variance weighting to reduce sensitivity to corrupted sensors
    while maintaining computational efficiency.
    """
    # slopes_multi shape: (n_wfs, n_slopes) or (n_wfs, n_channels, n_slopes_per_channel)
    
    # If slopes_multi has 3 dimensions, flatten channels dimension
    if slopes_multi.ndim == 3:
        n_wfs, n_channels, n_slopes = slopes_multi.shape
        slopes_flat = slopes_multi.reshape(n_wfs, n_channels * n_slopes)
    else:
        slopes_flat = slopes_multi
        n_wfs = slopes_multi.shape[0]
    
    # Compute local variance for each WFS channel
    # Use block-based variance estimation for robustness
    variances = np.var(slopes_flat, axis=1)
    
    # Add small epsilon to avoid division by zero
    variances = np.maximum(variances, 1e-6)
    
    # Compute inverse variance weights
    weights = 1.0 / variances
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Compute weighted average across WFS channels
    fused = np.zeros_like(slopes_flat[0])
    for i in range(n_wfs):
        fused += weights[i] * slopes_flat[i]
    
    # Apply reconstructor
    u = reconstructor @ fused
    
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
