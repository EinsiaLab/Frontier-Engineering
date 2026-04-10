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
    "ci_current": 2,      # current carbon intensity
    "temp_current": 4,    # current temperature metric
    "cooling_level": 6,   # current cooling setpoint (discrete)
}

BAT = {
    "ci_current": 2,          # carbon intensity seen by battery controller
    "soc": 5,                 # state of charge (0.0‑1.0)
    "power_demand": 7,        # current power draw of the data center
    "battery_power": 9,       # current battery power (positive if discharging)
    "charge_rate_limit": 11,  # max allowed charging rate
}


def reset_policy() -> None:
    """Reset any internal state between episodes."""
    return None


def _act_load_shifting(obs: Sequence[float]) -> int:
    """Decide load‑shifting action for the LS agent.

    Returns:
        0 – defer work to the future,
        1 – keep current schedule,
        2 – accelerate execution now.
    """
    # Extract relevant features.
    cur_ci = obs[LS["ci_current"]]
    fut_ci = obs[LS["ci_future_mean"]]
    q_fill = obs[LS["queue_fill_ratio"]]
    q_age = obs[LS["queue_oldest_age"]]
    workload = obs[LS["workload_current"]]
    overdue_share = obs[LS["queue_hist_over_24h"]]

    # 1️⃣ Emergency handling – overdue tasks or dangerously full/old queue.
    if overdue_share > 0.0 or q_age > 0.75 or q_fill > 0.90:
        return 2

    # 2️⃣ When future carbon is significantly greener, defer if we have slack.
    if (fut_ci - cur_ci) > 0.05 and q_fill < 0.70 and workload < 0.80:
        return 0

    # 3️⃣ When current carbon is noticeably cheaper, pull forward if queue permits.
    if (cur_ci - fut_ci) > 0.03 and q_fill > 0.15:
        return 2

    # 4️⃣ Moderate carbon difference – use queue pressure to decide.
    if cur_ci <= fut_ci and q_fill > 0.80:
        return 2
    if cur_ci >= fut_ci and q_fill < 0.30:
        return 0

    # 5️⃣ Default – stay neutral.
    return 1


def _act_cooling(obs: Sequence[float]) -> int:
    """Decide cooling level for the DC agent.

    Returns:
        0 – minimal cooling,
        1 – moderate cooling (default),
        2 – aggressive cooling.
    """
    if len(obs) <= max(DC.values()):
        return 1

    ci = obs[DC["ci_current"]]
    temp = obs[DC["temp_current"]]

    # High temperature forces aggressive cooling to protect hardware.
    if temp > 0.80:
        return 2

    # Very low temperature allows minimal cooling irrespective of carbon.
    if temp < 0.30:
        return 0

    # When carbon is expensive, we can trade a bit more water for lower IT load.
    if ci > 0.65:
        return 2 if temp > 0.55 else 1

    # When carbon is cheap, relax cooling aggressively to save water.
    if ci < 0.35:
        return 0 if temp < 0.55 else 1

    # Mid‑range carbon – choose cooling based on temperature band.
    if temp > 0.60:
        return 2
    if temp < 0.45:
        return 0
    return 1


def _act_battery(obs: Sequence[float]) -> int:
    """Decide battery action for the BAT agent.

    Returns:
        0 – discharge,
        1 – idle,
        2 – charge.
    """
    if len(obs) <= max(BAT.values()):
        return 1

    ci = obs[BAT["ci_current"]]
    soc = obs[BAT["soc"]]
    demand = obs[BAT["power_demand"]]
    batt_power = obs[BAT["battery_power"]]
    charge_limit = obs[BAT["charge_rate_limit"]]

    # Discharge aggressively when carbon is cheap, SOC is healthy, and load is high.
    if ci < 0.30 and soc > 0.40 and demand > 0.65:
        return 0

    # Secondary discharge rule – moderate carbon but high load and decent SOC.
    if ci < 0.40 and soc > 0.60 and demand > 0.55:
        return 0

    # Charge when carbon is cheap and we have capacity.
    if ci < 0.45 and soc < 0.90 and charge_limit > 0.0:
        # Prefer charging when load is low to avoid creating peaks.
        if demand < 0.55:
            return 2
        # If load is moderate but still below 0.70, allow charging.
        if demand < 0.70:
            return 2

    # Avoid charging when carbon is high; stay idle.
    if ci > 0.70:
        return 1

    # If battery is nearly empty, discharge even if carbon is moderate to keep services alive.
    if soc < 0.15 and demand > 0.30:
        return 0

    # Default idle.
    return 1


def decide_actions(observations: Mapping[str, Sequence[float]]) -> Dict[str, int]:
    """Map raw SustainDC observations to discrete actions for all agents."""
    return {
        "agent_ls": _act_load_shifting(observations["agent_ls"]),
        "agent_dc": _act_cooling(observations["agent_dc"]),
        "agent_bat": _act_battery(observations["agent_bat"]),
    }
# EVOLVE-BLOCK-END
