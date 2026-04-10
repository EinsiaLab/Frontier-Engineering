# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Dict, Mapping, Sequence

LS = {"ci": 2, "fci": 5, "age": 10, "fill": 12, "load": 13, "late": 25}

def reset_policy() -> None:
    """Reset any internal state between episodes.

    This baseline is stateless, so there is nothing to reset. The function is
    still provided so that users can add stateful logic later without changing
    the evaluation script.
    """


def _act_load_shifting(obs: Sequence[float]) -> int:
    ci, fci = obs[2], obs[5]
    age, fill, load, late = obs[10], obs[12], obs[13], obs[25]
    if late > 0 or age > 0.75 or fill > 0.8:
        return 2
    if ci < fci - 0.02 and fill > 0.05:
        return 2
    if ci > fci and fill < 0.95 and load < 0.9:
        return 0
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
        "agent_dc": 1,
        "agent_bat": 2,
    }
# EVOLVE-BLOCK-END
