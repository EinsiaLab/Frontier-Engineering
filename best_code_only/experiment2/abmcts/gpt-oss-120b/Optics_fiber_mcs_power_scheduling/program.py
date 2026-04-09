# EVOLVE-BLOCK-START
"""Baseline solver for Task 2: MCS + power scheduling."""

from __future__ import annotations

import numpy as np

# OptiCommPy provides a theoryBER function; use it if available.
try:
    from optic import theoryBER  # type: ignore
except Exception:  # pragma: no cover
    theoryBER = None  # fallback to simple thresholds


def _required_snr_db(mcs: int, target_ber: float = 1e-3) -> float:
    """
    Estimate the minimum SNR (in dB) required for the given MCS to achieve
    ``target_ber`` using ``optic.theoryBER``.  If the function is unavailable,
    a simple lookup table based on typical thresholds is used.
    """
    # Simple lookup for the three supported MCS values.
    # These thresholds are conservative and work even without theoryBER.
    lookup = {4: 10.0, 16: 15.0, 64: 22.0}
    if theoryBER is None:
        return lookup.get(mcs, lookup[64])

    # Search over Eb/N0 (dB) space.
    ebno_db = np.arange(0.0, 30.0, 0.1)
    for val in ebno_db:
        ber = theoryBER(mcs, val)
        if np.isnan(ber):
            continue
        if ber <= target_ber:
            return val
    # If not found, return a high value.
    return 30.0


def select_mcs_power(
    user_demands_gbps,
    channel_quality_db,
    total_power_dbm,
    mcs_candidates=(4, 16, 64),
    pmin_dbm=-8.0,
    pmax_dbm=4.0,
    target_ber=1e-3,
    seed=0,
):
    """
    Select modulation order (MCS) and launch power for each user.

    Parameters
    ----------
    user_demands_gbps : array‑like
        Desired throughput per user (Gb/s).
    channel_quality_db : array‑like
        Baseline channel quality metric (dB).  Higher values indicate a better
        channel (e.g., SNR without launch power).
    total_power_dbm : float
        Total launch power budget (dBm) shared among all users.
    mcs_candidates : tuple[int, ...], optional
        Available modulation orders.
    pmin_dbm, pmax_dbm : float, optional
        Per‑user power clipping bounds (dBm).
    target_ber : float, optional
        Target BER used for required‑SNR estimation.
    seed : int, optional
        Random seed (currently unused, kept for API compatibility).

    Returns
    -------
    dict
        ``{\"mcs\": np.ndarray, \"power_dbm\": np.ndarray}`` with shape ``(n_users,)``.
    """
    # ------------------------------------------------------------------
    # 0) Prepare inputs
    # ------------------------------------------------------------------
    demands = np.asarray(user_demands_gbps, dtype=float)
    quality = np.asarray(channel_quality_db, dtype=float)
    mcs_candidates = np.asarray(mcs_candidates, dtype=int)

    n_users = demands.size
    if n_users == 0:
        return {"mcs": np.array([], dtype=int), "power_dbm": np.array([], dtype=float)}

    # ------------------------------------------------------------------
    # 1) Power allocation (proportional to demand, with clipping & redistribution)
    # ------------------------------------------------------------------
    total_lin = 10.0 ** (float(total_power_dbm) / 10.0)  # total power in mW
    demand_sum = demands.sum()
    if demand_sum <= 0:
        raw_lin = np.full(n_users, total_lin / max(n_users, 1), dtype=float)
    else:
        raw_lin = total_lin * demands / demand_sum

    pmin_lin = 10.0 ** (pmin_dbm / 10.0)
    pmax_lin = 10.0 ** (pmax_dbm / 10.0)

    power_lin = np.clip(raw_lin, pmin_lin, pmax_lin)

    # Redistribute any leftover power uniformly while respecting per‑user max.
    leftover = total_lin - power_lin.sum()
    if leftover > 1e-12:
        can_gain = (pmax_lin - power_lin) > 1e-12
        while leftover > 1e-12 and np.any(can_gain):
            gain_per_user = leftover / np.count_nonzero(can_gain)
            increment = np.minimum(gain_per_user, pmax_lin - power_lin)
            power_lin[can_gain] += increment[can_gain]
            leftover = total_lin - power_lin.sum()
            can_gain = (pmax_lin - power_lin) > 1e-12

    power_dbm = 10.0 * np.log10(np.maximum(power_lin, 1e-12))
    power_dbm = np.clip(power_dbm, pmin_dbm, pmax_dbm)

    # ------------------------------------------------------------------
    # 2) Determine required SNR for each MCS (once per call)
    # ------------------------------------------------------------------
    required_snr = {
        int(m): _required_snr_db(int(m), target_ber=target_ber) for m in mcs_candidates
    }

    # ------------------------------------------------------------------
    # 3) MCS selection based on effective SNR (quality + launch power)
    # ------------------------------------------------------------------
    effective_snr_db = quality + power_dbm  # simple additive model

    # Start with the lowest MCS.
    mcs = np.full(n_users, int(mcs_candidates[0]), dtype=int)

    # Sort MCS candidates from low to high for deterministic upgrading.
    sorted_mcs = np.sort(mcs_candidates)
    for m in sorted_mcs[1:]:  # skip the lowest (already set)
        can_use = effective_snr_db >= required_snr[m]
        mcs[can_use] = int(m)

    return {"mcs": mcs, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
