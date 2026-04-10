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
    Slew-limited proportional control with temporal awareness.
    Clips command changes to respect actuator rate limits (reduces slew)
    while moving toward a scaled desired command from the reconstructor.
    Uses prev_commands for smoothness; tuned gain/slew for balance.
    """
    desired = reconstructor @ slopes
    # Moderate gain to avoid over-reaction to noisy/delayed slopes
    gain = control_model.get("gain", 0.65)
    desired = desired * gain

    # Slew limit close to ACTUATOR_RATE_LIMIT for smoothness
    slew_limit = control_model.get("slew_limit", 0.055)
    delta = np.clip(desired - prev_commands, -slew_limit, slew_limit)
    commands = prev_commands + delta

    return np.clip(commands, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
