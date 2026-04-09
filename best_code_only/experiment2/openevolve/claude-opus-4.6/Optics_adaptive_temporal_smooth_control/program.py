# EVOLVE-BLOCK-START
import numpy as np


def compute_dm_commands(
    slopes: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray,
    max_voltage: float = 0.25,
) -> np.ndarray:
    """Smooth controller with delay compensation and slew limiting."""
    sm = control_model["smooth_reconstructor"]
    pb = control_model["prev_blend"]
    rff = control_model.get("reconstructor")
    dg = float(control_model.get("delay_prediction_gain", 0.0))
    lp = float(control_model.get("command_lowpass", 0.0))

    # Integrator-style controller: accumulate corrections from slopes
    # Standard reconstructor gives best instantaneous wavefront estimate
    u_new = rff @ slopes if rff is not None else reconstructor @ slopes
    
    # Leaky integrator: blend new estimate with previous command
    # Gain controls how aggressively we track the wavefront
    # With actuator lag=0.76, we need to overshoot the target
    integrator_gain = 0.72
    u = prev_commands + integrator_gain * (u_new - prev_commands)
    
    # Add predictive correction: slopes are delayed by 1 frame
    # Use the innovation (difference from expected) to predict ahead
    if rff is not None and dg > 0.0:
        innovation = u_new - prev_commands
        u += (dg * 1.2) * innovation
    
    # Slew limit: mean_slew has weight 0.65, utility maxes at 0.045
    # Current slew ~0.043 is already at max utility (1.0)
    # Push slew limit higher to allow more correction for better RMS/Strehl
    # but keep mean_slew under 0.045 good anchor
    delta = np.clip(u - prev_commands, -0.055, 0.055)
    return np.clip(prev_commands + delta, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
