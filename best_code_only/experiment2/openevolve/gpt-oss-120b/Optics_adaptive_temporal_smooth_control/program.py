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
    Baseline: frame-wise independent control.

    Ignores temporal smoothness and command slew limits.
    """
    # -----------------------------------------------------------------
    # Analytical smooth controller (if matrices are provided)
    # -----------------------------------------------------------------
    # These matrices are supplied by the reference controller and give a
    # high‑quality estimate that already incorporates a smoothness term.
    smooth_reconstructor = control_model.get("smooth_reconstructor")
    prev_blend = control_model.get("prev_blend")
    reconstructor_ff = control_model.get("reconstructor")
    delay_gain = float(control_model.get("delay_prediction_gain", 0.0))
    lowpass = float(control_model.get("command_lowpass", 0.0))

    # Ensure we have a valid previous‑command vector.
    # `prev_commands` should have shape (n_act,).  If not, start from zero.
    if prev_commands is None or prev_commands.shape != reconstructor.shape[0]:
        prev = np.zeros(reconstructor.shape[0], dtype=np.float64)
    else:
        prev = prev_commands

    # Core estimate: use the analytical smooth matrices when available,
    # otherwise fall back to the plain reconstructor.
    if smooth_reconstructor is not None and prev_blend is not None:
        # u = smooth_reconstructor @ slopes + prev_blend @ prev
        u = smooth_reconstructor @ slopes + prev_blend @ prev
    else:
        u = reconstructor @ slopes

    # Optional delay‑aware feed‑forward correction (helps with lagged slopes).
    if reconstructor_ff is not None and delay_gain > 0.0:
        u += delay_gain * (reconstructor_ff @ slopes - prev)

    # Optional low‑pass blending with the previous command.
    if lowpass > 0.0:
        u = (1.0 - lowpass) * u + lowpass * prev

    # -----------------------------------------------------------------
    # Simple temporal blending & slew‑rate limiting (user‑tunable)
    # -----------------------------------------------------------------
    # `alpha` blends the newly computed command with the previous one.
    # The analytical controller already incorporates a low‑pass blend,
    # so an additional blend is unnecessary for most cases.
    # Default is set to 0 (no extra blending) but can be overridden via
    # `control_model["alpha"]` if the user wishes to experiment.
    alpha = float(control_model.get("alpha", 0.0))          # default: no extra blending
    # `max_slew` limits the per‑actuator voltage change between frames.
    # A slightly larger default allows the controller to react more
    # aggressively while still staying well within the hard rate‑limit
    # applied later in the evaluator.
    max_slew = float(control_model.get("max_slew", 0.07))   # volts per frame (relaxed)

    # Blend with previous command.
    u = alpha * prev + (1.0 - alpha) * u

    # Apply hard slew‑rate limit.
    delta = u - prev
    delta = np.clip(delta, -max_slew, max_slew)
    u = prev + delta

    # Finally enforce the hardware voltage limits
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
