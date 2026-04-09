#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    # 4-stage CC profile tuned for speed vs. safe V/T/plating margins
    return {
        "currents_c": [5.7, 4.1, 2.5, 1.05],
        "switch_soc": [0.42, 0.68, 0.79],
    }


# EVOLVE-BLOCK-END
