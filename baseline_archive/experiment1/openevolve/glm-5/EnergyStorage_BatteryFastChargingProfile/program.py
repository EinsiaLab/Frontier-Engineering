#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    """Optimized multi-stage fast-charge profile.
    
    Pushes second-stage current to 5.9C with extended high-current phase
    at switch point 0.48. Uses remaining voltage headroom (~0.06V) and
    thermal headroom (~16C) to minimize charge time while staying safe.
    """
    return {
        "currents_c": [6.0, 5.9, 4.0, 2.5],
        "switch_soc": [0.48, 0.69, 0.78],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
