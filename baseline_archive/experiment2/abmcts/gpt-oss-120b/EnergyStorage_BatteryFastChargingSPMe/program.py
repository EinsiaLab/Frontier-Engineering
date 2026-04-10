#!/usr/bin/env python3
# EVOLVE-BLOCK-START
from __future__ import annotations


def build_charging_policy() -> dict:
    """
    Returns a deterministic staged charging policy that aims to reduce total
    charging time while staying within the hard voltage, temperature, and
    plating‑margin limits defined in the benchmark configuration.

    The policy consists of four current stages (in C‑rate) and three SOC
    switching thresholds.  The thresholds are chosen to keep the higher
    currents active for a larger portion of the charge, which shortens the
    overall charge time without exceeding safety limits in the simulated
    SPMe‑T‑Aging model.
    """
    return {
        # Higher currents for longer portions of the charge, still within limits.
        "currents_c": [3.6, 2.9, 2.2, 1.4],
        # Switch later so the higher currents are applied over a wider SOC range.
        "switch_soc": [0.30, 0.65, 0.85],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
