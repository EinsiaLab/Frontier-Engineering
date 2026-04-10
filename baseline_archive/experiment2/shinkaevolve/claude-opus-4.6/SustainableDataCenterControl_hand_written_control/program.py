# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Dict, Mapping, Sequence


# ---------------------------------------------------------------------------
# Observation indices
# ---------------------------------------------------------------------------

# agent_ls observations (26 features):
#   0: month_sin, 1: month_cos, 2: ci_current, 3: ci_future_1, 4: ci_future_2,
#   5: ci_future_mean, 6: temp_current, 7: temp_future_1, 8: temp_future_2,
#   9: temp_future_mean, 10: queue_oldest_age, 11: queue_avg_age,
#   12: queue_fill_ratio, 13: workload_current, 14-25: queue_age_histogram
LS = {
    "ci_current": 2,
    "ci_future_1": 3,
    "ci_future_2": 4,
    "ci_future_mean": 5,
    "temp_current": 6,
    "queue_oldest_age": 10,
    "queue_avg_age": 11,
    "queue_fill_ratio": 12,
    "workload_current": 13,
    "queue_hist_over_24h": 25,
}

# agent_dc observations (14 features):
DC = {
    "ci_current": 2,
    "ci_future_mean": 3,
    "temp_outdoor": 4,
    "it_load_fraction": 6,
    "cooling_setpoint": 7,
    "indoor_temp": 8,
    "cooling_power": 9,
    "total_power": 10,
    "water_usage": 11,
    "carbon_footprint": 12,
    "pue": 13,
}

# agent_bat observations (13 features):
BAT = {
    "ci_current": 2,
    "ci_future_1": 3,
    "ci_future_2": 4,
    "ci_future_mean": 5,
    "soc": 8,
    "energy_price": 9,
    "grid_power": 10,
    "net_load": 11,
    "carbon_footprint": 12,
}

# ---------------------------------------------------------------------------
# Internal state for adaptive CI thresholds
# ---------------------------------------------------------------------------
_ci_ema: float = 0.5
_ci_ema_initialized: bool = False
_step_count: int = 0


def reset_policy() -> None:
    """Reset any internal state between episodes."""
    global _ci_ema, _ci_ema_initialized, _step_count
    _ci_ema = 0.5
    _ci_ema_initialized = False
    _step_count = 0


def _act_load_shifting(obs: Sequence[float]) -> int:
    """Load shifting: defer work when grid is dirty, execute when clean.

    Action 0: defer (don't execute flexible tasks now)
    Action 1: normal processing
    Action 2: aggressive execution (drain backlog)
    """
    global _ci_ema, _ci_ema_initialized

    current_ci = obs[LS["ci_current"]]
    future_ci_mean = obs[LS["ci_future_mean"]]
    future_ci_1 = obs[LS["ci_future_1"]]
    future_ci_2 = obs[LS["ci_future_2"]]
    queue_fill = obs[LS["queue_fill_ratio"]]
    oldest_age = obs[LS["queue_oldest_age"]]
    workload = obs[LS["workload_current"]]
    overdue_share = obs[LS["queue_hist_over_24h"]]

    # Update EMA of CI for context
    alpha = 0.05
    if not _ci_ema_initialized:
        _ci_ema = current_ci
        _ci_ema_initialized = True
    else:
        _ci_ema = alpha * current_ci + (1 - alpha) * _ci_ema

    # ---- EMERGENCY: must drain queue ----
    if overdue_share > 0.0:
        return 2
    if oldest_age > 0.60:
        return 2
    if queue_fill > 0.75:
        return 2

    # Safety: aging queue needs attention
    if oldest_age > 0.45 and queue_fill > 0.25:
        return 2

    # ---- CI-ratio based decisions ----
    ci_ratio = current_ci / (future_ci_mean + 1e-9)

    # Use minimum of future forecasts for stronger deferral signal
    future_min = min(future_ci_1, future_ci_2, future_ci_mean)
    ci_vs_min = current_ci / (future_min + 1e-9)

    # ---- DEFER: current CI is high relative to future ----

    # Very strong deferral: much dirtier than best upcoming
    if ci_vs_min > 1.20 and queue_fill < 0.55 and oldest_age < 0.42:
        return 0

    # Strong deferral
    if ci_ratio > 1.08 and queue_fill < 0.48 and oldest_age < 0.40:
        return 0

    # Moderate deferral
    if ci_ratio > 1.03 and queue_fill < 0.38 and oldest_age < 0.32 and workload < 0.85:
        return 0

    # Mild deferral
    if ci_ratio > 1.005 and queue_fill < 0.25 and oldest_age < 0.22 and workload < 0.75:
        return 0

    # ---- EXECUTE: current CI is low relative to future ----

    # Very aggressive execution
    if ci_ratio < 0.84 and queue_fill > 0.003:
        return 2

    # Aggressive execution
    if ci_ratio < 0.91 and queue_fill > 0.008:
        return 2

    # Moderate execution
    if ci_ratio < 0.96 and queue_fill > 0.03:
        return 2

    # Mild execution
    if ci_ratio < 0.993 and queue_fill > 0.10:
        return 2

    # ---- QUEUE MANAGEMENT ----
    if queue_fill > 0.22 and ci_ratio <= 1.005:
        return 2

    if oldest_age > 0.30 and queue_fill > 0.08:
        return 2

    if queue_fill > 0.28:
        return 2

    return 1


def _act_cooling(obs: Sequence[float]) -> int:
    """Cooling control: nearly pure noop to avoid water penalty.

    Action 0: lower setpoint (more cooling - AVOID: increases water dramatically)
    Action 1: maintain setpoint (noop - safest for water)
    Action 2: raise setpoint (less cooling - saves energy but can increase water)

    Key insight from ALL prior experiments: ANY deviation from noop increases
    water usage by 3-4% across all scenarios. The carbon savings from cooling
    changes are small compared to the water penalty. Stay noop except for
    true temperature emergencies.
    """
    indoor_temp = obs[DC["indoor_temp"]]

    # TRUE EMERGENCY ONLY: indoor temp dangerously high
    if indoor_temp > 0.85:
        return 0

    # Everything else: noop. Water savings >> carbon savings from cooling shifts.
    return 1


def _act_battery(obs: Sequence[float]) -> int:
    """Battery dispatch: store clean energy, release during dirty periods.

    Action 0: charge (adds grid load NOW - increases carbon NOW)
    Action 1: idle
    Action 2: discharge (offsets grid load NOW - decreases carbon NOW)

    Strategy: More aggressive cycling. The avg_soc of 0.039 in prior runs
    shows we're barely using the battery. We need to charge more when clean
    and discharge more when dirty to maximize carbon arbitrage.
    """
    ci_current = obs[BAT["ci_current"]]
    ci_future = obs[BAT["ci_future_mean"]]
    ci_future_1 = obs[BAT["ci_future_1"]]
    ci_future_2 = obs[BAT["ci_future_2"]]
    soc = obs[BAT["soc"]]

    ci_ratio = ci_current / (ci_future + 1e-9)

    # Use max of future CIs for charge decisions
    future_max = max(ci_future, ci_future_1, ci_future_2)
    ci_vs_max = ci_current / (future_max + 1e-9)

    # ---- CHARGE: grid is clean now, will be dirty later ----
    # Charge aggressively when current CI is much lower than future max
    if ci_vs_max < 0.70 and soc < 0.98:
        return 0

    if ci_vs_max < 0.80 and soc < 0.90:
        return 0

    if ci_vs_max < 0.88 and soc < 0.80:
        return 0

    if ci_ratio < 0.85 and soc < 0.85:
        return 0

    if ci_ratio < 0.92 and soc < 0.65:
        return 0

    if ci_ratio < 0.96 and soc < 0.45:
        return 0

    if ci_ratio < 0.99 and soc < 0.25:
        return 0

    # ---- DISCHARGE: grid is dirty now ----
    if ci_ratio > 1.20 and soc > 0.05:
        return 2  # discharge - very dirty

    if ci_ratio > 1.12 and soc > 0.10:
        return 2  # discharge

    if ci_ratio > 1.06 and soc > 0.20:
        return 2  # discharge

    if ci_ratio > 1.02 and soc > 0.40:
        return 2  # discharge if reasonably charged

    if ci_ratio > 1.00 and soc > 0.55:
        return 2  # discharge if well charged, even slightly dirty

    # If battery is very full, discharge to make room
    if soc > 0.82:
        return 2

    # Default: idle to preserve charge
    if soc > 0.08:
        return 1

    # Default: discharge (noop equivalent)
    return 2


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