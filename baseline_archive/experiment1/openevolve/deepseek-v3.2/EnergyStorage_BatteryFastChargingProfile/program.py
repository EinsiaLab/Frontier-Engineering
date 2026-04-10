#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    """Optimized multi-stage constant-current fast-charge profile."""
    # Inspired by top performer: 3-stage approach for faster charging
    # Adjust switch points to reduce charge time while maintaining safety
    # Extend the second stage to SOC 0.77 to reduce time, while keeping currents unchanged
    return {
        "currents_c": [6.0, 4.5, 2.0],
        "switch_soc": [0.45, 0.77],
    }


def main() -> None:
    profile = build_charging_profile()
    print(profile)


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
