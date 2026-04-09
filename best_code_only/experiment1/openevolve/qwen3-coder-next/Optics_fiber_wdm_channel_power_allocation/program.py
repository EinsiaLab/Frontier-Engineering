# EVOLVE-BLOCK-START
"""WDM channel + power allocation solver.

Allocates channels to users based on demand matching and power efficiency.
"""

from __future__ import annotations
import numpy as np


def allocate_wdm(
    user_demands_gbps,
    channel_centers_hz,
    total_power_dbm,
    pmin_dbm=-8.0,
    pmax_dbm=3.0,
    target_ber=1e-3,
    seed=0,
):
    """Allocate channels and power to maximize demand satisfaction."""
    demands = np.asarray(user_demands_gbps, dtype=float)
    n_users, n_channels = len(demands), len(channel_centers_hz)
    
    # Sort users by demand (higher demand first) for better allocation
    sorted_idx = np.argsort(-demands)
    assignment = -np.ones(n_users, dtype=int)
    power_dbm = np.full(n_channels, pmin_dbm, dtype=float)
    
    # Greedy assignment: match high-demand users to channels with interference awareness
    n_served = min(n_users, n_channels)
    if n_served > 0:
        # Create evenly spaced channel indices for better isolation
        channel_indices = np.linspace(0, n_channels - 1, n_served).astype(int)
        # Sort channels to be more centralized for better performance
        center = (n_channels - 1) / 2.0
        channel_indices = channel_indices[np.argsort(np.abs(channel_indices - center))]
        
        # Assign high-demand users to the best channels
        for i in range(n_served):
            assignment[sorted_idx[i]] = channel_indices[i]
    
    used = np.unique(assignment[assignment >= 0])
    if used.size == 0:
        return {"assignment": assignment, "power_dbm": power_dbm}
    
    # Power allocation: proportional to demand with budget enforcement
    total_power_lin = 10 ** (float(total_power_dbm) / 10.0)
    used_demands = demands[assignment >= 0]
    demand_ratio = used_demands / used_demands.sum()
    
    # Initial power allocation in linear domain
    ch_power_lin = np.zeros(n_channels)
    for idx, ch in enumerate(used):
        ch_power_lin[ch] = total_power_lin * demand_ratio[idx]
    
    # Apply interference-aware power adjustment
    # Channels with better isolation (fewer neighbors) get slightly more power
    for idx, ch in enumerate(used):
        # Count nearby active channels (potential interferers)
        nearby_count = 0
        for other_ch in used:
            if other_ch != ch and abs(other_ch - ch) <= 2:  # Within 2 channels
                nearby_count += 1
        
        # Slightly increase power for better-isolated channels (less interference)
        # and reduce power for crowded channels
        if nearby_count == 0:
            ch_power_lin[ch] *= 1.10  # 10% boost for isolated channels
        elif nearby_count == 1:
            ch_power_lin[ch] *= 1.05  # 5% boost for channels with one neighbor
        else:
            ch_power_lin[ch] *= max(0.90, 1.0 - 0.05 * nearby_count)  # Reduce for crowded
    
    # Renormalize to respect total power budget while preserving relative adjustments
    total_lin = np.sum(ch_power_lin)
    if total_lin > 0 and total_lin > total_power_lin * 1.001:
        scale = total_power_lin / total_lin
        ch_power_lin *= scale
    
    # Clip to power limits and convert to dBm
    ch_power_dbm = 10 * np.log10(np.maximum(ch_power_lin, 1e-12))
    ch_power_dbm = np.clip(ch_power_dbm, pmin_dbm, pmax_dbm)
    
    # Final power adjustment to ensure budget constraint is met exactly
    current_total_lin = np.sum(10 ** (ch_power_dbm / 10.0))
    if current_total_lin > total_power_lin * 1.001:  # Allow small numerical tolerance
        # Scale down powers proportionally while respecting bounds
        scale_factor = total_power_lin / current_total_lin
        ch_power_lin_scaled = 10 ** (ch_power_dbm / 10.0) * scale_factor
        ch_power_dbm = 10 * np.log10(np.maximum(ch_power_lin_scaled, 1e-12))
        ch_power_dbm = np.clip(ch_power_dbm, pmin_dbm, pmax_dbm)
    
    power_dbm = ch_power_dbm
    
    return {"assignment": assignment, "power_dbm": power_dbm}
# EVOLVE-BLOCK-END
