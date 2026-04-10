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
    Enhanced control with temporal smoothing and delay compensation.
    
    Uses precomputed control matrices from control_model for smooth control:
    - smooth_reconstructor: for spatial smoothing
    - prev_blend: for temporal smoothing with previous commands
    - delay_prediction_gain: for feed-forward delay compensation
    - command_lowpass: for additional temporal filtering
    - explicit slew rate limiting for better temporal smoothness
    """
    # Extract control parameters from control_model
    smooth_reconstructor = control_model.get("smooth_reconstructor", reconstructor)
    prev_blend = control_model.get("prev_blend", np.eye(len(prev_commands)))
    reconstructor_ff = control_model.get("reconstructor", reconstructor)
    delay_prediction_gain = float(control_model.get("delay_prediction_gain", 0.0))
    command_lowpass = float(control_model.get("command_lowpass", 0.0))
    # Extract max slew rate from control_model or use default
    max_slew = float(control_model.get("max_slew", 0.055)) if control_model else 0.055
    
    # Skip slope noise filtering as it may add unnecessary complexity
    # Focus on core temporal smoothing and delay compensation
    
    # Base smooth command from precomputed matrices
    u = smooth_reconstructor @ slopes + prev_blend @ prev_commands
    
    # Apply delay-aware feed-forward correction
    if delay_prediction_gain > 0.0:
        u += delay_prediction_gain * (reconstructor_ff @ slopes - prev_commands)
    
    # Apply optional low-pass blending with previous command
    if command_lowpass > 0.0:
        u = (1.0 - command_lowpass) * u + command_lowpass * prev_commands
    
    # Apply adaptive gain scheduling based on command magnitude
    # Reduce gain for larger commands to avoid overshooting
    command_magnitude = np.max(np.abs(u))
    if command_magnitude > max_voltage * 0.7:
        # Reduce gain when approaching limits to prevent overshoot
        # More aggressive gain reduction for better stability
        gain_reduction = max(0.85, 1.0 - (command_magnitude - max_voltage * 0.7) / (max_voltage * 0.3))
        u = u * gain_reduction
    
    # Apply explicit slew rate limiting for better temporal smoothness
    # Calculate actual change from previous command
    delta_u = u - prev_commands
    # Clip the change to max_slew for better temporal smoothness
    delta_u_clipped = np.clip(delta_u, -max_slew, max_slew)
    
    # Apply limited change
    u = prev_commands + delta_u_clipped
    
    # Final voltage limiting with soft clipping for smoother transitions
    # Use small margin to avoid hard clipping effects
    margin = 0.005
    u = np.tanh(u / (max_voltage - margin)) * (max_voltage - margin)
    
    return u
# EVOLVE-BLOCK-END
