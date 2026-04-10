# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Dict, Mapping, Sequence


# Observation indices for the three agents.
LS = {
    "ci_current": 2,
    "ci_future_mean": 5,
    "queue_oldest_age": 10,
    "queue_fill_ratio": 12,
    "workload_current": 13,
    "queue_hist_over_24h": 25,
}

# Data center agent indices (based on benchmark_core.py DC_FEATURES)
DC = {
    "ci_current": 2,           # ci_current_norm
    "workload_current": 10,    # workload_current
    "temp_current": 12,        # outdoor_temp_current_norm
    "temp_next": 13,           # outdoor_temp_next_norm
}

# Battery agent indices (based on benchmark_core.py BAT_FEATURES)
BAT = {
    "soc": 12,                 # battery_soc
    "ci_current": 2,           # ci_current_norm
    "ci_future_mean": 5,       # ci_future_mean
}

def reset_policy() -> None:
    """Reset any internal state between episodes.

    This baseline is stateless, so there is nothing to reset. The function is
    still provided so that users can add stateful logic later without changing
    the evaluation script.
    """


def _act_load_shifting(obs: Sequence[float]) -> int:
    current_ci = obs[LS["ci_current"]]
    future_ci = obs[LS["ci_future_mean"]]
    queue_fill = obs[LS["queue_fill_ratio"]]
    oldest_age = obs[LS["queue_oldest_age"]]
    workload = obs[LS["workload_current"]]
    overdue_share = obs[LS["queue_hist_over_24h"]]

    # Emergency queue draining if tasks are too old or the queue is almost full.
    # Use thresholds from the top performer (Program 1)
    if overdue_share > 0.05 or oldest_age > 0.75 or queue_fill > 0.80:
        return 2

    # When the future is cleaner than now, save flexible work for later.
    if current_ci > future_ci and queue_fill < 0.95 and workload < 0.90:
        return 0

    # When now is cleaner than the near future and some backlog exists, execute it.
    if current_ci < future_ci - 0.02 and queue_fill > 0.05:
        return 2

    return 1


def _act_data_center(obs: Sequence[float]) -> int:
    """Decide data center action (0: decrease cooling, 1: unchanged, 2: increase cooling)."""
    current_ci = obs[DC["ci_current"]]
    workload = obs[DC["workload_current"]]
    temp_current = obs[DC["temp_current"]]
    
    # Use the refined policy from the top performer (Program 1)
    # When carbon is high (>0.65), use less cooling (action 2) to save energy
    # When carbon is low (<0.3), we can use more cooling (action 0) if needed
    if current_ci > 0.65:  # High carbon intensity
        return 2  # Increase setpoint (less cooling)
    elif current_ci < 0.3:  # Low carbon intensity
        # Check if workload or temperature is high to decide if we need more cooling
        if workload > 0.7 or temp_current > 0.5:
            # High workload or temperature, maintain current cooling
            return 1
        else:
            # Conditions favorable for more cooling
            return 0  # Decrease setpoint (more cooling)
    else:
        return 1  # Keep unchanged


def _act_battery(obs: Sequence[float]) -> int:
    """Decide battery action (0: charge, 1: idle, 2: discharge)."""
    soc = obs[BAT["soc"]]
    current_ci = obs[BAT["ci_current"]]
    future_ci = obs[BAT["ci_future_mean"]]
    
    # Use the enhanced policy from the top performer (Program 1)
    # Charge when carbon is low now but will get dirtier, and battery is not too full
    if current_ci < 0.4 and future_ci > current_ci + 0.05 and soc < 0.85:
        return 0  # charge
    # Discharge when carbon is high now but will get cleaner, and battery has enough charge
    elif current_ci > 0.6 and future_ci < current_ci - 0.05 and soc > 0.15:
        return 2  # discharge
    # Otherwise, stay idle
    else:
        return 1  # idle


def decide_actions(observations: Mapping[str, Sequence[float]]) -> Dict[str, int]:
    """Map raw SustainDC observations to discrete actions.

    Args:
        observations: A dictionary with three entries:
            - observations["agent_ls"]: shape (26,)
            - observations["agent_dc"]: shape (14,)
            - observations["agent_bat"]: shape (13,)

    Returns:
        A dictionary with one discrete action per agent.
    """

    return {
        "agent_ls": _act_load_shifting(observations["agent_ls"]),
        "agent_dc": _act_data_center(observations["agent_dc"]),
        "agent_bat": _act_battery(observations["agent_bat"]),
    }
# EVOLVE-BLOCK-END
