# EVOLVE-BLOCK-START
from __future__ import annotations

def solve() -> dict:
    d = [120.0, 90.0, 60.0, 40.0, 25.0, 18.0, 12.0, 8.0]
    t = (2.0 * (100.0 + 40.0 + 35.0 + 30.0 + 28.0 + 25.0 + 22.0 + 20.0 + 18.0)
         / (1.8 * 120.0 + 2.0 * 90.0 + 1.6 * 60.0 + 1.7 * 40.0 + 1.5 * 25.0 + 1.9 * 18.0 + 2.1 * 12.0 + 1.4 * 8.0)) ** 0.5
    return {"base_cycle_time": t, "order_multiples": [1] * 8, "order_quantities": [x * t for x in d]}
# EVOLVE-BLOCK-END
