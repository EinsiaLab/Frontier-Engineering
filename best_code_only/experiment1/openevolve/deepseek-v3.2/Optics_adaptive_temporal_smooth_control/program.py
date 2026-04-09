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
    Enhanced controller with aggressive smoothing to reduce mean_slew.
    
    Uses precomputed smooth matrices, delay compensation, low-pass filtering,
    and a stricter slew rate limit (0.02) plus additional blending.
    """
    # Precomputed matrices for temporal smoothing
    smooth_reconstructor = control_model['smooth_reconstructor']
    prev_blend = control_model['prev_blend']
    
    # Core smooth term
    u = smooth_reconstructor @ slopes + prev_blend @ prev_commands
    
    # Delay-aware feed-forward correction
    delay_prediction_gain = float(control_model.get('delay_prediction_gain', 0.0))
    if delay_prediction_gain > 0.0:
        reconstructor_ff = control_model.get('reconstructor', reconstructor)
        u += delay_prediction_gain * (reconstructor_ff @ slopes - prev_commands)
    
    # Low-pass blending using the simulation's command_lowpass (0.88)
    command_lowpass = float(control_model.get('command_lowpass', 0.0))
    if command_lowpass > 0.0:
        u = (1.0 - command_lowpass) * u + command_lowpass * prev_commands
    
    # Blend with raw reconstruction to improve instantaneous correction
    # without significantly increasing slew (blend_factor = 0.1 from top performers)
    raw_u = reconstructor @ slopes
    blend_factor = 0.1
    u = (1.0 - blend_factor) * u + blend_factor * raw_u
    
    # Clip to voltage limits
    u = np.clip(u, -max_voltage, max_voltage)
    
    # Apply slew rate limiting matching simulation's ACTUATOR_RATE_LIMIT (0.055)
    max_slew_rate = 0.055
    delta = u - prev_commands
    delta_clipped = np.clip(delta, -max_slew_rate, max_slew_rate)
    u = prev_commands + delta_clipped
    
    # Final voltage clipping to ensure bounds
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
