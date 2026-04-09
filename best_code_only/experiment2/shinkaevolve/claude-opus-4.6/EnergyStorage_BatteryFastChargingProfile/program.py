#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    """Aggressive multi-stage constant-current fast-charge profile.

    Pushes high C-rates in low-SOC region where voltage headroom is large,
    then steps down progressively to manage voltage and plating near target SOC.
    """
    return {
        "currents_c": [6.0, 4.5, 3.5, 2.5, 1.8, 1.2],
        "switch_soc": [0.35, 0.50, 0.60, 0.70, 0.76],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END