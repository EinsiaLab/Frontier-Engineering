# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Dict, Mapping, Sequence


def reset_policy() -> None:
    pass


def _act_load_shifting(obs: Sequence[float]) -> int:
    if obs[25] > 0.0 or obs[10] > 0.80 or obs[12] > 0.85:
        return 2

    if obs[2] > obs[5] and obs[12] < 0.90 and obs[13] < 0.90:
        return 0

    if obs[2] < obs[5] - 0.01 and obs[12] > 0.05:
        return 2

    return 1


def decide_actions(observations: Mapping[str, Sequence[float]]) -> Dict[str, int]:
    dc, bat = observations["agent_dc"], observations["agent_bat"]
    
    a_dc = 2 if dc[2] > dc[5] + 0.03 and dc[12] < 0.6 else (0 if dc[2] < dc[5] - 0.03 and dc[12] > 0.4 else 1)
    a_bat = 0 if bat[2] < bat[5] - 0.03 and bat[12] < 0.90 else (1 if bat[2] > bat[5] + 0.03 and bat[12] > 0.10 else 2)

    return {
        "agent_ls": _act_load_shifting(observations["agent_ls"]),
        "agent_dc": a_dc,
        "agent_bat": a_bat,
    }
# EVOLVE-BLOCK-END
