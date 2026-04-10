# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Dict, Mapping, Sequence


def reset_policy() -> None:
    """Reset any internal state between episodes (stateless)."""


def _act_load_shifting(obs: Sequence[float]) -> int:
    current_ci = obs[2]
    future_ci = obs[5]
    queue_fill = obs[12]
    oldest_age = obs[10]
    workload = obs[13]
    overdue_share = obs[25]
    temp = obs[14]

    # Emergency queue draining if tasks are too old or the queue is almost full.
    if overdue_share > 0.0 or oldest_age > 0.75 or queue_fill > 0.80:
        return 2

    # Avoid peak workload during hot periods (helps carbon/water in summer scenarios).
    if workload > 0.84 and temp > 0.72 and queue_fill < 0.92:
        return 0

    # When the future is cleaner than now, save flexible work for later.
    if current_ci > future_ci and queue_fill < 0.95 and workload < 0.90:
        return 0

    # When now is cleaner than the near future and some backlog exists, execute it.
    if current_ci < future_ci - 0.02 and queue_fill > 0.05:
        return 2

    return 1


def _act_cooling(obs: Sequence[float]) -> int:
    ci = obs[2]
    temp = obs[12]
    workload = obs[10]
    next_temp = obs[13]
    next_load = obs[11]
    # Slightly higher thresholds reduce unnecessary cooling (water) while
    # still protecting against thermal pressure; higher ci threshold for
    # relaxing cooling avoids carbon penalty in winter.
    if (temp > 0.76 or next_temp > 0.76 or workload > 0.86 or next_load > 0.86):
        return 0
    if ci > 0.73 and workload < 0.67:
        return 2
    return 1


def _act_battery(obs: Sequence[float]) -> int:
    ci_now = obs[2]
    ci_future = obs[5]
    ci_pct = obs[7]
    soc = obs[12]
    # Tighter charging conditions avoid increasing grid draw unless very
    # advantageous; slightly adjusted discharge helps winter carbon.
    if ci_now < 0.36 and ci_future > ci_now + 0.04 and soc < 0.72 and ci_pct < 0.43:
        return 0
    if (ci_now > 0.67 or ci_pct > 0.74) and soc > 0.23:
        return 1
    return 2


def decide_actions(observations: Mapping[str, Sequence[float]]) -> Dict[str, int]:
    """Map observations to actions using tuned threshold rules."""
    return {
        "agent_ls": _act_load_shifting(observations["agent_ls"]),
        "agent_dc": _act_cooling(observations["agent_dc"]),
        "agent_bat": _act_battery(observations["agent_bat"]),
    }
# EVOLVE-BLOCK-END
