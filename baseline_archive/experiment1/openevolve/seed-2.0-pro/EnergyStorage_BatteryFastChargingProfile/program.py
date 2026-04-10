#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    """Optimized multi-stage constant-current fast-charge profile balancing charge speed and soft voltage limit compliance to maximize total combined score, while maintaining zero plating loss, low aging, and all hard safety constraints."""
    return {
        "currents_c": [6.0, 5.2, 3.8, 2.4],
        "switch_soc": [0.46, 0.70, 0.78],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
