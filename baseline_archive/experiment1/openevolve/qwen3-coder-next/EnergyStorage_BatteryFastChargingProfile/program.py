#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    """Optimized multi-stage constant-current fast-charge profile with improved throughput."""
    return {
        "currents_c": [6.0, 4.8, 3.5, 2.6, 1.8, 1.0],
        "switch_soc": [0.20, 0.45, 0.65, 0.73, 0.785],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
