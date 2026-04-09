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

BAT = {
    "soc": 0,
    "ci_current": 2,
    "ci_future_mean": 5,
}

DC = {
    "outdoor_temp": 0,
    "indoor_temp": 1,
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

    # Emergency queue draining if tasks are too old or queue is critically full.
    if overdue_share > 0.0 or oldest_age > 0.75 or queue_fill > 0.85:
        return 2

    # When the future is cleaner than now, save flexible work for later.
    if current_ci > future_ci and queue_fill < 0.95 and workload < 0.90:
        return 0

    # When now is meaningfully cleaner than the near future, execute backlog.
    # Lowered threshold to 0.01 for more aggressive clean energy execution.
    if current_ci < future_ci - 0.01 and queue_fill > 0.02:
        return 2

    # Execute during very clean energy periods (CI < 0.18) with meaningful work.
    # Lowered threshold to capture more clean energy opportunities.
    if current_ci < 0.18 and queue_fill > 0.02:
        return 2

    return 1


def _act_battery(obs: Sequence[float]) -> int:
    """Battery dispatch strategy: charge during clean periods, discharge during dirty."""
    soc = obs[BAT["soc"]]
    current_ci = obs[BAT["ci_current"]]
    future_ci = obs[BAT["ci_future_mean"]]

    # Charge during clean periods - more selective for truly clean energy.
    # CI < 0.22 ensures charging during genuinely clean periods, SOC < 0.95 maximizes storage.
    if current_ci < 0.22 and soc < 0.95:
        return 0

    # Discharge during very dirty periods - maximize carbon offset.
    # CI > 0.70 for effective carbon offset, SOC > 0.08 preserves minimum reserve.
    if current_ci > 0.70 and soc > 0.08:
        return 2

    # If future is cleaner than now and battery is low, wait to charge later.
    if future_ci < current_ci - 0.04 and soc < 0.75:
        return 1

    # If future is dirtier than now and battery has charge, discharge now.
    if future_ci > current_ci + 0.04 and soc > 0.15:
        return 2

    return 1


def _act_cooling(obs: Sequence[float]) -> int:
    """Cooling strategy: leverage free cooling when outdoor temperature is low."""
    outdoor_temp = obs[DC["outdoor_temp"]]
    indoor_temp = obs[DC["indoor_temp"]]

    # When outdoor temp is low enough, use aggressive cooling (lower setpoint)
    # for free cooling. Lower threshold to 12°C to maximize water savings.
    if outdoor_temp < 12.0:
        return 0

    # When outdoor temp is moderately warm, raise setpoint to reduce cooling load
    # and water consumption. Lower threshold to 26°C for earlier intervention.
    if outdoor_temp > 26.0:
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

    return {
        "agent_ls": _act_load_shifting(observations["agent_ls"]),
        "agent_dc": _act_cooling(observations["agent_dc"]),
        "agent_bat": _act_battery(observations["agent_bat"]),
    }
# EVOLVE-BLOCK-END