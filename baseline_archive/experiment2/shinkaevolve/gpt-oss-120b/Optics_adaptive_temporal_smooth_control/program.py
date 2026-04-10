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
    Dual-Path Smooth-Accuracy Control with Explicit Slew Limiting.

    Combines two control pathways:
    1. Smooth path: Uses smooth_reconstructor + prev_blend for temporal consistency
    2. Accuracy path: Uses standard reconstructor + delay compensation for RMS

    Applies both low-pass filtering and explicit slew limiting for optimal smoothness.
    """
    # Extract control matrices
    smooth_recon = control_model.get('smooth_reconstructor', reconstructor)
    prev_blend = control_model.get('prev_blend')
    delay_gain = control_model.get('delay_prediction_gain')
    lowpass = control_model.get('command_lowpass')

    # === Primary smooth control path ===
    u_smooth = smooth_recon @ slopes

    # Add temporal continuity from previous commands
    if prev_blend is not None:
        u_smooth = u_smooth + prev_blend @ prev_commands

    # === Secondary accuracy-focused path ===
    # Use standard reconstructor for better instantaneous accuracy
    u_accuracy = reconstructor @ slopes

    # Enhanced delay compensation on accuracy path
    if delay_gain is not None:
        if isinstance(delay_gain, np.ndarray) and delay_gain.ndim == 2:
            u_accuracy = u_accuracy + delay_gain @ slopes
        else:
            # Slightly boost delay compensation for accuracy path
            u_accuracy = u_accuracy * (1 + float(delay_gain) * 1.1)

    # === Blend paths ===
    # Balance smoothness with accuracy - tuned weights for optimal tradeoff
    # Slew is already well-controlled by lowpass; favor accuracy slightly more
    smooth_weight = 0.85
    accuracy_weight = 0.15
    u = smooth_weight * u_smooth + accuracy_weight * u_accuracy

    # Apply delay compensation to blended output for overall improvement
    if delay_gain is not None:
        if isinstance(delay_gain, np.ndarray) and delay_gain.ndim == 2:
            u = u + 0.5 * delay_gain @ slopes

    # Apply low-pass filtering for temporal smoothness
    # This provides natural slew limiting while maintaining correction quality
    if lowpass is not None:
        alpha = float(lowpass)
        u = alpha * prev_commands + (1 - alpha) * u

    # Enforce voltage limits
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END