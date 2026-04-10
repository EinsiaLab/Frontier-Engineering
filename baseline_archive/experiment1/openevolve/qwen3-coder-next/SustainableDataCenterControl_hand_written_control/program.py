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
    if overdue_share > 0.0 or oldest_age > 0.75 or queue_fill > 0.80:
        return 2

    # When the future is cleaner than now, save flexible work for later.
    if current_ci > future_ci and queue_fill < 0.95 and workload < 0.90:
        return 0

    # When now is cleaner than the near future and some backlog exists, execute it.
    if current_ci < future_ci - 0.02 and queue_fill > 0.05:
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

    ls_action = _act_load_shifting(observations["agent_ls"])
    
    # Datacenter agent uses queue information to decide
    dc_obs = observations["agent_dc"]
    dc_queue_fill = dc_obs[12] if len(dc_obs) > 12 else 0.0
    dc_workload = dc_obs[13] if len(dc_obs) > 13 else 0.0
    
    # Datacenter: prioritize when workload is low and queue is manageable
    dc_action = 1  # default
    if dc_workload < 0.6 and dc_queue_fill < 0.7:
        dc_action = 0  # save work for cleaner energy
    elif dc_workload > 0.8 or dc_queue_fill > 0.9:
        dc_action = 2  # execute or drain
    
    # Battery agent makes decisions based on current CI and workload
    bat_obs = observations["agent_bat"]
    bat_ci = bat_obs[2] if len(bat_obs) > 2 else 0.5
    bat_workload = bat_obs[12] if len(bat_obs) > 12 else 0.5
    
    # Battery: charge when dirty energy, discharge when clean
    bat_action = 2 if bat_ci > 0.5 else 0  # default charge/discharge
    
    return {
        "agent_ls": ls_action,
        "agent_dc": dc_action,
        "agent_bat": bat_action,
    }
# EVOLVE-BLOCK-END
