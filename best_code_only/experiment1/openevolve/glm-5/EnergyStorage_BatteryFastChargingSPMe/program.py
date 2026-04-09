#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    # Aggressive 4-stage profile - higher mid-currents, later transitions
    # Stage 1: 4.5C to 0.20 SOC - push initial current higher
    # Stage 2: 4.0C to 0.50 SOC - maintain high current longer
    # Stage 3: 3.0C to 0.75 SOC - stronger mid-stage current
    # Stage 4: 1.8C to target - controlled finish
    return {
        "currents_c": [4.5, 4.0, 3.0, 1.8],
        "switch_soc": [0.20, 0.50, 0.75],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
