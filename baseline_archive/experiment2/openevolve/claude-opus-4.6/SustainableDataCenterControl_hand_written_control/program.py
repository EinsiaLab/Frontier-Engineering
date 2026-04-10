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

DC = {
    "ci_current": 2,
    "ci_future_mean": 5,
    "ci_percentile": 7,
    "workload": 10,
    "workload_next": 11,
    "outside_temp": 12,
    "outside_temp_next": 13,
}

BAT = {
    "ci_current": 2,
    "ci_future_slope": 3,
    "ci_future_mean": 5,
    "ci_percentile": 7,
    "workload": 10,
    "outside_temp": 11,
    "battery_soc": 12,
}

def reset_policy() -> None:
    """Reset any internal state between episodes.

    This baseline is stateless, so there is nothing to reset. The function is
    still provided so that users can add stateful logic later without changing
    the evaluation script.
    """


def _act_load_shifting(obs: Sequence[float]) -> int:
    current_ci = obs[2]
    future_ci = obs[5]
    oldest_age = obs[10]
    queue_fill = obs[12]
    workload = obs[13]
    overdue_share = obs[25]

    # Emergency: drain queue if tasks are overdue or queue is critically full.
    if overdue_share > 0.0 or oldest_age > 0.65 or queue_fill > 0.75:
        return 2

    # Defer when future is cleaner and queue has room.
    ci_ratio = current_ci / (future_ci + 1e-8)
    if ci_ratio > 1.05 and queue_fill < 0.60 and workload < 0.85:
        return 0

    # Execute when now is cleaner than future and there's queued work.
    if ci_ratio < 0.95 and queue_fill > 0.02:
        return 2

    # Moderate deferral if slightly dirtier now and queue isn't building up.
    if current_ci > future_ci and queue_fill < 0.40:
        return 0

    return 1


def _act_dc_cooling(obs: Sequence[float]) -> int:
    ci = obs[2]
    if ci > 0.6:
        return 2  # Higher setpoint = less cooling energy when grid is dirty
    elif ci < 0.3:
        return 0  # Lower setpoint = more cooling but CI is low
    return 1


def _act_battery(obs: Sequence[float]) -> int:
    ci = obs[2]
    future_ci = obs[5]
    soc = obs[12]

    # Charge when grid is clean and battery isn't full.
    if ci < 0.35 and soc < 0.85:
        return 0
    # When grid is dirty and battery has charge, go idle (avoid grid draw).
    if ci > 0.55 and soc > 0.20:
        return 2  # Idle is safer than discharge
    # Charge if future will be dirtier and battery has room.
    if future_ci > ci * 1.1 and soc < 0.70:
        return 0
    return 1  # Default idle-ish


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
        "agent_dc": _act_dc_cooling(observations["agent_dc"]),
        "agent_bat": _act_battery(observations["agent_bat"]),
    }
# EVOLVE-BLOCK-END
