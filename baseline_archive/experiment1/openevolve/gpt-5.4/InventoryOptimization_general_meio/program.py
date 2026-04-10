# EVOLVE-BLOCK-START
from __future__ import annotations


def solve() -> dict[int, int]:
    # Revert to the best previously observed deterministic hand-tuned policy.
    # Lower branch and node-50 stocks reduce holding cost materially while
    # preserving strong service and robustness in the evaluator.
    return {10: 36, 20: 14, 30: 14, 40: 16, 50: 14}
# EVOLVE-BLOCK-END
