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
    Robust sensor fusion with outlier detection and temporal smoothing.
    
    Uses multiple robust strategies to detect and downweight corrupted
    sensor channels, then applies optimized leaky integrator temporal smoothing.
    """
    n_sensors = slopes_multi.shape[0]
    
    if n_sensors <= 1:
        fused = slopes_multi[0]
        u = reconstructor @ fused
        if prev_commands is not None:
            gain = control_model.get('gain', 0.4)
            leak = control_model.get('leak', 0.9)
            u = leak * prev_commands + gain * u
        return np.clip(u, -max_voltage, max_voltage)
    
    if n_sensors == 2:
        fused = np.median(slopes_multi, axis=0)
        u = reconstructor @ fused
        if prev_commands is not None:
            gain = control_model.get('gain', 0.4)
            leak = control_model.get('leak', 0.9)
            u = leak * prev_commands + gain * u
        return np.clip(u, -max_voltage, max_voltage)
    
    # Step 1: Compute per-channel statistics for outlier detection
    # Use the median slope vector as robust reference
    median_slopes = np.median(slopes_multi, axis=0)
    
    # Compute deviation of each sensor from the median
    deviations = slopes_multi - median_slopes[np.newaxis, :]
    
    # Per-sensor RMS deviation from median
    rms_devs = np.sqrt(np.mean(deviations**2, axis=1))
    
    # Robust scale estimate using MAD of the rms deviations
    median_rms = np.median(rms_devs)
    mad_rms = np.median(np.abs(rms_devs - median_rms))
    if mad_rms < 1e-12:
        mad_rms = np.std(rms_devs) + 1e-12
    
    # Modified z-scores for each sensor
    z_scores = 0.6745 * (rms_devs - median_rms) / (mad_rms + 1e-12)
    
    # Step 2: Also compute per-element correlation-based score
    # Sensors that correlate poorly with the median are likely corrupted
    median_norm = np.linalg.norm(median_slopes)
    if median_norm > 1e-12:
        corr_scores = np.array([
            np.dot(slopes_multi[i], median_slopes) / (np.linalg.norm(slopes_multi[i]) * median_norm + 1e-12)
            for i in range(n_sensors)
        ])
    else:
        corr_scores = np.ones(n_sensors)
    
    # Step 3: Compute weights combining both metrics
    # Z-score based weights (soft thresholding with tighter threshold)
    threshold = 1.5
    w_zscore = np.exp(-np.maximum(z_scores - threshold, 0.0)**2 / 1.5)
    w_zscore[z_scores > 4.0] = 0.0
    
    # Correlation-based weights (penalize anti-correlated or uncorrelated sensors)
    w_corr = np.clip(corr_scores, 0.0, 1.0) ** 2
    
    # Combined weights
    weights = w_zscore * w_corr
    
    # Identify clear inliers: keep only top sensors if separation is clear
    sorted_idx = np.argsort(rms_devs)
    if n_sensors >= 4:
        # Check if there's a gap between inliers and outliers
        sorted_rms = rms_devs[sorted_idx]
        gaps = np.diff(sorted_rms)
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            if gaps[max_gap_idx] > 2.0 * np.median(sorted_rms[:max_gap_idx+1] + 1e-12):
                # Clear separation - zero out outliers
                outlier_indices = sorted_idx[max_gap_idx+1:]
                weights[outlier_indices] = 0.0
    
    # Ensure at least some sensors contribute
    if np.sum(weights) < 1e-12:
        # Fall back to using the sensor closest to median
        best = np.argmin(rms_devs)
        weights = np.zeros(n_sensors)
        weights[best] = 1.0
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Step 4: Weighted fusion
    fused = np.sum(slopes_multi * weights[:, np.newaxis], axis=0)
    
    # Step 5: Compute DM commands
    u = reconstructor @ fused
    
    # Step 6: Temporal smoothing (leaky integrator with tuned parameters)
    if prev_commands is not None:
        gain = control_model.get('gain', 0.4)
        leak = control_model.get('leak', 0.88)
        u = leak * prev_commands + gain * u
    
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
