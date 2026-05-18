# EVOLVE-BLOCK-START
from __future__ import annotations

from typing import Dict, Mapping, Sequence

LS = 2, 5, 10, 12, 13, 25
BAT = 2, 5, 12

def reset_policy() -> None: return None


def _act_load_shifting(obs: Sequence[float]) -> int:
    c, f, a, q, w, o = (obs[i] for i in LS)
    if o > 0.0 or a > 0.80 or q > 0.85:
        return 2
    if c > f + 0.01 and w < 0.90:
        return 0
    return 2 if c < f - 0.02 and q > 0.05 else 1





def _act_battery(obs: Sequence[float]) -> int:
    c, f, s = (obs[i] for i in BAT)
    return 1 if s > 0.35 and (c > f + 0.03 or s > 0.65 and c > 0.45) else 2


def decide_actions(observations: Mapping[str, Sequence[float]]) -> Dict[str, int]:
    return {
        "agent_ls": _act_load_shifting(observations["agent_ls"]),
        "agent_dc": 1,
        "agent_bat": _act_battery(observations["agent_bat"]),
    }
# EVOLVE-BLOCK-END
