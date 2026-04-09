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
    Temporal smooth control: uses control_model matrices for
    smooth reconstruction with delay compensation and low-pass filtering.

    Tuned to trade excess smoothness margin for better RMS/Strehl,
    with enhanced delay compensation for high actuator lag scenarios.
    """
    # Tuning parameters
    # We have massive slew margin (0.002 vs 0.045 threshold) so push correction hard
    lowpass_scale = 0.02  # Nearly disable lowpass - we have huge slew margin
    # Boost delay prediction to aggressively compensate actuator lag (0.76)
    delay_boost = 2.8
    # Overall gain to push correction much harder for better RMS/Strehl
    overall_gain = 1.45

    # Extract control matrices from the model
    smooth_recon = control_model.get("smooth_reconstructor", None)
    prev_blend = control_model.get("prev_blend", None)
    delay_gain = control_model.get("delay_prediction_gain", None)
    lowpass = control_model.get("command_lowpass", None)

    # Step 1: Main reconstruction with smoothness
    if smooth_recon is not None and prev_blend is not None:
        u = smooth_recon @ slopes + prev_blend @ prev_commands
    else:
        u = reconstructor @ slopes

    # Apply overall gain boost to improve correction strength
    u = overall_gain * u + (1.0 - overall_gain) * prev_commands

    # Step 2: Delay prediction compensation (boosted)
    if delay_gain is not None:
        diff = u - prev_commands
        # delay_gain could be a scalar or a matrix
        if np.isscalar(delay_gain) or (isinstance(delay_gain, np.ndarray) and delay_gain.ndim == 0):
            # Scalar gain: apply as prediction correction
            u = u + float(delay_gain) * delay_boost * diff
        elif isinstance(delay_gain, np.ndarray) and delay_gain.ndim == 2:
            # Matrix gain
            u = u + delay_boost * (delay_gain @ diff)
        elif isinstance(delay_gain, np.ndarray) and delay_gain.ndim == 1:
            # Vector gain (element-wise)
            u = u + delay_boost * delay_gain * diff

    # Step 3: Low-pass filtering with previous commands (reduced effect)
    if lowpass is not None:
        if np.isscalar(lowpass) or (isinstance(lowpass, np.ndarray) and lowpass.ndim == 0):
            alpha = float(lowpass) * lowpass_scale
            u = alpha * prev_commands + (1.0 - alpha) * u
        elif isinstance(lowpass, np.ndarray) and lowpass.ndim == 1:
            scaled_lp = lowpass * lowpass_scale
            u = scaled_lp * prev_commands + (1.0 - scaled_lp) * u
        elif isinstance(lowpass, np.ndarray) and lowpass.ndim == 2:
            scaled_lp = lowpass * lowpass_scale
            u = scaled_lp @ prev_commands + (np.eye(scaled_lp.shape[0]) - scaled_lp) @ u

    # Step 4: Clip to voltage bounds
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END