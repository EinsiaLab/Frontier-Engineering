#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    return {
        # Revert to the best previously observed feasible schedule:
        # - stronger early charging where plating margin is generous
        # - more gradual taper through mid/high SOC
        # - stays below the soft voltage limit, which materially boosts score
        "currents_c": [4.3, 3.5, 2.8, 1.9, 1.2],
        "switch_soc": [0.18, 0.42, 0.68, 0.84],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
