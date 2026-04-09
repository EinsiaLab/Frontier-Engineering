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
    Temporal‑smooth adaptive optics controller.

    The baseline implementation only applied a frame‑wise reconstructor.
    This version adds simple yet effective smoothness handling using the
    information supplied in ``control_model`` and the previous command
    vector.

    Parameters
    ----------
    slopes : np.ndarray
        Measured wave‑front slopes (shape ``(n_slopes,)``).
    reconstructor : np.ndarray
        Conventional reconstructor matrix (shape ``(n_actuators, n_slopes)``).
    control_model : dict
        Optional model parameters that can contain:
        * ``smooth_reconstructor`` – pre‑computed matrix that already
          includes a smoothness term.
        * ``prev_blend`` – matrix that blends the previous command
          (e.g. ``I - α·K`` where ``K`` is a gain matrix).
        * ``blend`` – scalar in ``[0, 1]`` for simple convex blending
          between the new command and ``prev_commands`` (default 0.5).
        * ``lowpass`` – scalar in ``[0, 1]`` for an additional low‑pass
          filter on the command update (default ``None`` → no extra
          filtering).
        * ``delay_gain`` – scalar gain applied to a delay‑prediction term.
    prev_commands : np.ndarray
        Command vector from the previous time step (shape
        ``(n_actuators,)``). If ``None`` a zero vector is assumed.
    max_voltage : float, optional
        Voltage clipping limit (default ``0.25``).

    Returns
    -------
    np.ndarray
        Voltage commands for the deformable mirror, clipped to
        ``[-max_voltage, max_voltage]``.
    """
    # Ensure a valid previous command vector
    if prev_commands is None:
        prev_commands = np.zeros(reconstructor.shape[0], dtype=slopes.dtype)

    # 1️⃣  Base reconstruction – allow a smooth‑aware matrix if provided
    if "smooth_reconstructor" in control_model:
        u = control_model["smooth_reconstructor"] @ slopes
    else:
        u = reconstructor @ slopes

    # 2️⃣  Add a contribution that explicitly uses the previous command
    if "prev_blend" in control_model:
        # Matrix‑based blending (e.g. I - α·K)
        u += control_model["prev_blend"] @ prev_commands
    else:
        # Simple scalar convex combination
        blend = float(control_model.get("blend", 0.5))
        u = blend * u + (1.0 - blend) * prev_commands

    # 3️⃣  Optional delay‑prediction feed‑forward term
    if "delay_gain" in control_model:
        gain = float(control_model["delay_gain"])
        u += gain * prev_commands  # crude prediction of the next command

    # 4️⃣  Optional low‑pass filter on the update
    lowpass = control_model.get("lowpass", None)
    if lowpass is not None:
        lp = float(lowpass)
        u = lp * u + (1.0 - lp) * prev_commands

    # 5️⃣  Clip to the physical voltage limits
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
