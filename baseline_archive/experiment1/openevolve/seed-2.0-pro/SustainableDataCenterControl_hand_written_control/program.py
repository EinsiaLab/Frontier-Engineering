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
    if overdue_share > 0.0 or oldest_age > 0.75 or queue_fill > 0.85:
        return 2

    # When the future is cleaner than now, save flexible work for later.
    if current_ci > future_ci - 0.005 and queue_fill < 0.96 and workload < 0.92:
        return 0

    # When now is cleaner than the near future and some backlog exists, execute it.
    if current_ci < future_ci - 0.015 and queue_fill > 0.03:
        return 2

    return 1


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

    # Battery optimization: charge when CI is low, discharge when CI is high
    bat_obs = observations["agent_bat"]
    bat_current_ci = bat_obs[2]
    bat_future_ci = bat_obs[5]
    bat_soc = bat_obs[3]  # State of charge (0 = empty, 1 = full)
    bat_action = 1  # Default: hold charge
    if bat_current_ci < bat_future_ci - 0.008 and bat_soc < 0.97:
        bat_action = 0  # Charge even for small future CI increases, nearly fill battery
    elif bat_current_ci > bat_future_ci + 0.008 and bat_soc > 0.03:
        bat_action = 2  # Discharge even for small future CI decreases, nearly empty battery

    # DC cooling optimization: adjust based on workload to save energy
    dc_obs = observations["agent_dc"]
    dc_workload = dc_obs[4]
    dc_action = 1  # Default: normal cooling
    if dc_workload > 0.88:
        dc_action = 2  # Increase cooling earlier at high workload for safety
    elif dc_workload < 0.35:
        dc_action = 0  # Reduce cooling for more low-workload scenarios to save energy

    return {
        "agent_ls": _act_load_shifting(observations["agent_ls"]),
        "agent_dc": dc_action,
        "agent_bat": bat_action,
    }
# EVOLVE-BLOCK-END
