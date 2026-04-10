#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    """Aggressive multi-stage constant-current fast-charge profile."""
    return {
        "currents_c": [6.0, 5.45, 4.24, 2.74],
        "switch_soc": [0.41, 0.65, 0.78],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
