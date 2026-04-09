#!/usr/bin/env python3
# EVOLVE-BLOCK-START
from __future__ import annotations


def build_charging_profile() -> dict:
    """Optimized multi‑stage constant‑current fast‑charge profile.

    Uses higher early currents to reduce total charge time while staying
    safely below the hard voltage (4.25 V) and temperature (47 °C) limits.
    """
    return {
        "currents_c": [6.0, 5.0, 4.0, 3.0, 2.0],
        "switch_soc": [0.30, 0.55, 0.70, 0.78],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
