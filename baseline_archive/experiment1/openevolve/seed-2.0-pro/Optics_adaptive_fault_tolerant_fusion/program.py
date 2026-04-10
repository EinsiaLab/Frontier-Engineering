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
    Baseline: naive average over all WFS channels.

    Sensitive to corrupted sensors.
    """
    # Use anomaly-aware weighted fusion when model is available, fallback to robust median otherwise
    model = control_model.get("anomaly_model")
    if model is None or slopes_multi.shape[0] < 2:
        fused = np.median(slopes_multi, axis=0)
    else:
        # Get fusion hyperparameters from control model if present
        # Match system default inlier fraction for optimal outlier rejection / noise balance
        inlier_fraction = float(control_model.get("inlier_fraction", 0.4))
        score_temperature = float(control_model.get("score_temperature", 0.08))
        # Score each WFS channel for normality using pre-trained anomaly detector
        channel_scores = model.decision_function(slopes_multi)
        # Keep only top-scoring inlier channels (filter out corrupted ones)
        n_keep = max(1, int(np.ceil(inlier_fraction * slopes_multi.shape[0])))
        keep_idx = np.argsort(channel_scores)[-n_keep:]
        # Median of top inlier channels for maximum robustness to any remaining outlier noise
        fused = np.median(slopes_multi[keep_idx], axis=0)
    
    # Optional delay compensation if h_matrix is provided in control model
    if prev_commands is not None and control_model.get("h_matrix") is not None:
        delay_comp_gain = float(control_model.get("delay_comp_gain", 0.0))
        if delay_comp_gain > 0.0:
            fused = fused + delay_comp_gain * (control_model["h_matrix"] @ prev_commands)
    
    # Ensure fused slopes are strictly finite to avoid invalid command outputs
    fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
    u = reconstructor @ fused
    
    # Add optional temporal smoothing with previous commands for stable control
    if prev_commands is not None:
        # Use evaluator-provided temporal blend parameter if available, clamp to valid [0,1] range
        temporal_blend = float(np.clip(control_model.get("temporal_blend", 0.05), 0.0, 1.0))
        u = (1.0 - temporal_blend) * u + temporal_blend * prev_commands
    
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
