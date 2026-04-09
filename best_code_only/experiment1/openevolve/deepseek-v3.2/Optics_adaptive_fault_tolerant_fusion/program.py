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
    Robust fusion combining anomaly scores and distance-based reliability.
    Dynamically adjusts inlier selection based on reliability spread.
    """
    n_wfs = slopes_multi.shape[0]
    
    # Compute median as robust reference for distance measure
    median_slopes = np.median(slopes_multi, axis=0)
    distances = np.sqrt(np.mean((slopes_multi - median_slopes) ** 2, axis=1))
    
    # Normalize distances to [0,1] range
    if np.max(distances) > 0:
        distances_norm = distances / np.max(distances)
    else:
        distances_norm = np.zeros_like(distances)
    
    # Use anomaly model if available
    model = control_model.get('anomaly_model')
    if model is None or n_wfs < 2:
        # Fallback: use inverse distance as reliability
        reliability = 1.0 - distances_norm
    else:
        anomaly_scores = model.decision_function(slopes_multi)
        # Normalize anomaly scores to [0,1] range
        anomaly_scores = anomaly_scores - np.min(anomaly_scores)
        if np.max(anomaly_scores) > 0:
            anomaly_scores = anomaly_scores / np.max(anomaly_scores)
        
        # Combine anomaly scores and distances into composite reliability score
        # Higher anomaly score and lower distance indicate more reliable
        reliability = anomaly_scores * (1.0 - distances_norm)
    
    # Determine number of inliers dynamically based on score spread
    score_spread = np.max(reliability) - np.min(reliability)
    base_inlier_fraction = control_model.get('inlier_fraction', 0.4)
    if score_spread < 0.1:  # Not discriminative
        n_keep = max(1, int(np.ceil(base_inlier_fraction * n_wfs)))
    else:
        # More discriminative: keep fewer channels
        n_keep = max(1, int(np.ceil(base_inlier_fraction * n_wfs * 0.5)))
    
    # Select top reliable channels
    keep_idx = np.argsort(reliability)[-n_keep:]
    
    # Weighted fusion using softmax on reliability scores of kept channels
    rel_kept = reliability[keep_idx]
    rel_kept = rel_kept - np.max(rel_kept)  # normalize for stability
    temperature = control_model.get('score_temperature', 0.08)
    weights = np.exp(rel_kept / (temperature + 1e-12))
    weights = weights / (np.sum(weights) + 1e-12)
    fused = np.sum(slopes_multi[keep_idx] * weights[:, np.newaxis], axis=0)
    
    u = reconstructor @ fused
    
    # Temporal blending using parameter from control_model if available
    if prev_commands is not None:
        temporal_blend = control_model.get('temporal_blend', 0.0)
        if temporal_blend > 0.0:
            u = (1.0 - temporal_blend) * u + temporal_blend * prev_commands
    
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
