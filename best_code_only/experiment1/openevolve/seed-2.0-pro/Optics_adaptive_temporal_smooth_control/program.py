# EVOLVE-BLOCK-START
import numpy as np


def compute_dm_commands(
    slopes: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray,
    max_voltage: float = 0.25,
) -> np.ndarray:
    """
    Improved control with temporal smoothing, slew limiting, and gain adjustment.
    
    Uses previous commands to enforce physical DM constraints and reduce noise.
    """
    # Precompute base reconstruction to avoid redundant matrix multiplication
    recon_slopes = reconstructor @ slopes
    # Use precomputed smooth reconstructor for balanced error/smoothness tradeoff
    smooth_recon = control_model.get("smooth_reconstructor", reconstructor)
    u_raw = smooth_recon @ slopes
    
    # Add delay compensation to correct for stale WFS measurements (optimized gain for lag correction)
    delay_gain = control_model.get("delay_prediction_gain", 0.57)
    u_raw += delay_gain * (recon_slopes - prev_commands)
    
    # Optimized gain for correction without overshoot
    gain = control_model.get("control_gain", 0.95)
    u_gained = gain * u_raw
    
    # Balanced smoothing factor: reduced for better dynamic turbulence response without excessive slew
    smoothing_factor = control_model.get("smoothing_factor", 0.76)
    u_smoothed = smoothing_factor * u_gained + (1 - smoothing_factor) * prev_commands
    
    # Enforce maximum allowed voltage change per frame, slightly increased to allow larger corrective steps
    max_slew = control_model.get("max_slew_per_step", 0.058)
    delta = np.clip(u_smoothed - prev_commands, -max_slew, max_slew)
    u_slew_limited = prev_commands + delta
    
    # Final clamp to maximum allowed DM voltage range
    return np.clip(u_slew_limited, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
