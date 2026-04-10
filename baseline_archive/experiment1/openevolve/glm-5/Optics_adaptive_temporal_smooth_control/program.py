# EVOLVE-BLOCK-START
import numpy as np

def compute_dm_commands(slopes, reconstructor, control_model, prev_commands, max_voltage=0.25):
    """Integrator with optimal slew limiting for RMS/slew tradeoff."""
    delta = np.clip(0.72 * (reconstructor @ slopes), -0.055, 0.055)
    return np.clip(prev_commands + delta, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
