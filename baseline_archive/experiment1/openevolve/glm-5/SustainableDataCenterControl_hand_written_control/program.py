# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Dict, Mapping, Sequence




def reset_policy() -> None:
    """Reset any internal state between episodes.

    This baseline is stateless, so there is nothing to reset. The function is
    still provided so that users can add stateful logic later without changing
    the evaluation script.
    """


def _act_load_shifting(obs: Sequence[float]) -> int:
    ci, ci_fut = obs[2], obs[5]
    q_fill, q_age, work = obs[12], obs[10], obs[13]
    over_24 = obs[25]
    
    # Emergency draining for overdue tasks or near-full queue
    if over_24 > 0.0 or q_age > 0.55 or q_fill > 0.80:
        return 2
    
    ratio = ci / max(ci_fut, 0.001)
    
    # Future cleaner: defer tasks (lower threshold for more deferral)
    if ratio > 1.12 and q_fill < 0.70 and work < 0.75:
        return 0
    
    # Now cleaner: execute backlog
    if ratio < 0.92 and q_fill > 0.02:
        return 2
    
    return 1


def _act_battery(obs: Sequence[float]) -> int:
    """Battery: keep idle to avoid increasing grid draw.
    
    Battery operations consume energy during charging/discharging.
    Actions: 0=charge, 1=discharge, 2=idle
    """
    return 2


def _act_dc(obs: Sequence[float]) -> int:
    """Conservative cooling: more cooling during peak hours."""
    if obs[0] > 0.55 and obs[1] < 0.35: return 0
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
        "agent_dc": _act_dc(observations["agent_dc"]),
        "agent_bat": _act_battery(observations["agent_bat"]),
    }
# EVOLVE-BLOCK-END
