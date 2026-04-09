# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Dict, Mapping, Sequence

# ---------------------------------------------------------------------------
# Observation indices for the three agents.
# ---------------------------------------------------------------------------
# agent_ls observation space (26 features):
#   0: month_sin, 1: month_cos
#   2: ci_current (normalized carbon intensity now)
#   3: ci_future_1h, 4: ci_future_2h, 5: ci_future_mean
#   6: outside_temp, 7: outside_rh
#   8: queue_length, 9: queue_total_age
#   10: queue_oldest_age, 11: queue_newest_age
#   12: queue_fill_ratio, 13: workload_current
#   14-24: queue histogram bins, 25: queue_hist_over_24h

LS = {
    "ci_current": 2,
    "ci_future_1h": 3,
    "ci_future_2h": 4,
    "ci_future_mean": 5,
    "outside_temp": 6,
    "queue_oldest_age": 10,
    "queue_fill_ratio": 12,
    "workload_current": 13,
    "queue_hist_over_24h": 25,
}

# agent_dc observation space (14 features):
#   0: month_sin, 1: month_cos
#   2: ci_current, 3: outside_temp, 4: outside_rh
#   5: it_power, 6: hvac_power, 7: total_power
#   8: cooling_setpoint, 9: avg_zone_temp
#   10: cpu_util, 11: supply_temp, 12: return_temp, 13: water_usage

DC = {
    "ci_current": 2,
    "outside_temp": 3,
    "outside_rh": 4,
    "it_power": 5,
    "hvac_power": 6,
    "total_power": 7,
    "cooling_setpoint": 8,
    "avg_zone_temp": 9,
    "cpu_util": 10,
    "supply_temp": 11,
    "return_temp": 12,
    "water_usage": 13,
}

# agent_bat observation space (13 features):
#   0: month_sin, 1: month_cos
#   2: ci_current, 3: ci_future_1h, 4: ci_future_2h, 5: ci_future_mean
#   6: outside_temp, 7: outside_rh
#   8: battery_soc, 9: net_energy
#   10: it_power, 11: hvac_power, 12: total_power

BAT = {
    "ci_current": 2,
    "ci_future_1h": 3,
    "ci_future_2h": 4,
    "ci_future_mean": 5,
    "battery_soc": 8,
    "net_energy": 9,
    "it_power": 10,
    "hvac_power": 11,
    "total_power": 12,
}

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------
_step_count = 0
_ci_history = []

def reset_policy() -> None:
    """Reset any internal state between episodes."""
    global _step_count, _ci_history
    _step_count = 0
    _ci_history = []


def _act_load_shifting(obs: Sequence[float]) -> int:
    current_ci = obs[LS["ci_current"]]
    future_ci = obs[LS["ci_future_mean"]]
    future_1h = obs[LS["ci_future_1h"]]
    future_2h = obs[LS["ci_future_2h"]]
    queue_fill = obs[LS["queue_fill_ratio"]]
    oldest_age = obs[LS["queue_oldest_age"]]
    workload = obs[LS["workload_current"]]
    overdue_share = obs[LS["queue_hist_over_24h"]]

    # Emergency: drain queue immediately if tasks are getting old or queue is very full
    if overdue_share > 0.0:
        return 2
    if oldest_age > 0.65:
        return 2
    if queue_fill > 0.75:
        return 2

    # Strong signal: current CI much higher than future → defer aggressively
    ci_ratio = current_ci / (future_ci + 1e-8)
    
    if ci_ratio > 1.3 and queue_fill < 0.55 and oldest_age < 0.45:
        return 0  # defer strongly
    
    if ci_ratio > 1.1 and queue_fill < 0.50 and oldest_age < 0.40 and workload < 0.85:
        return 0  # defer moderately
    
    # Current is cleaner than future → execute more now
    if current_ci < future_ci * 0.85 and queue_fill > 0.02:
        return 2  # execute aggressively
    
    if current_ci < future_ci * 0.95 and queue_fill > 0.10:
        return 2  # execute moderately

    # Moderate deferral when current is slightly dirtier
    if current_ci > future_ci * 1.05 and queue_fill < 0.40 and oldest_age < 0.35:
        return 0

    return 1


def _act_cooling(obs: Sequence[float]) -> int:
    """
    DC cooling setpoint control.
    Action 0 = lower setpoint (more cooling, more energy, less water stress)
    Action 1 = keep current setpoint
    Action 2 = raise setpoint (less cooling, less energy, but risk overheating)
    
    Strategy: raise setpoint when CI is high to save energy/carbon,
    lower when CI is low and temps are comfortable.
    """
    ci = obs[DC["ci_current"]]
    outside_temp = obs[DC["outside_temp"]]
    avg_zone_temp = obs[DC["avg_zone_temp"]]
    cpu_util = obs[DC["cpu_util"]]
    hvac_power = obs[DC["hvac_power"]]
    water_usage = obs[DC["water_usage"]]
    
    # If zone temp is getting high, don't raise setpoint further
    if avg_zone_temp > 0.75:
        return 0  # lower setpoint to cool down
    
    # If CI is high and temps are comfortable, raise setpoint to save energy
    if ci > 0.6 and avg_zone_temp < 0.55 and cpu_util < 0.7:
        return 2  # raise setpoint
    
    if ci > 0.5 and avg_zone_temp < 0.45:
        return 2  # raise setpoint
    
    # If CI is low and we have thermal headroom concerns, lower setpoint
    if ci < 0.3 and avg_zone_temp > 0.6:
        return 0
    
    # If water usage is high, consider raising setpoint (less cooling = less water)
    if water_usage > 0.7 and avg_zone_temp < 0.50:
        return 2
    
    return 1


def _act_battery(obs: Sequence[float]) -> int:
    """
    Battery control.
    Action 0 = charge (consume grid energy now)
    Action 1 = hold / idle
    Action 2 = discharge (offset grid energy now)
    
    Strategy: charge when CI is low, discharge when CI is high.
    """
    ci = obs[BAT["ci_current"]]
    future_ci = obs[BAT["ci_future_mean"]]
    future_1h = obs[BAT["ci_future_1h"]]
    soc = obs[BAT["battery_soc"]]
    
    # Discharge when current CI is high and we have charge
    if ci > 0.6 and soc > 0.25:
        return 2  # discharge
    
    if ci > 0.5 and ci > future_ci * 1.15 and soc > 0.20:
        return 2  # discharge - current is dirtier than future
    
    # Charge when current CI is low and battery not full
    if ci < 0.3 and soc < 0.85:
        return 0  # charge
    
    if ci < 0.4 and ci < future_ci * 0.80 and soc < 0.75:
        return 0  # charge - current is cleaner than future
    
    # Moderate: charge if current much cleaner than future
    if ci < future_ci * 0.75 and soc < 0.70:
        return 0
    
    # Moderate: discharge if current much dirtier than future
    if ci > future_ci * 1.25 and soc > 0.30:
        return 2
    
    return 1  # idle


def decide_actions(observations: Mapping[str, Sequence[float]]) -> Dict[str, int]:
    """Map raw SustainDC observations to discrete actions."""
    global _step_count
    _step_count += 1

    obs_ls = observations["agent_ls"]
    obs_dc = observations["agent_dc"]
    obs_bat = observations["agent_bat"]

    return {
        "agent_ls": _act_load_shifting(obs_ls),
        "agent_dc": _act_cooling(obs_dc),
        "agent_bat": _act_battery(obs_bat),
    }
# EVOLVE-BLOCK-END
