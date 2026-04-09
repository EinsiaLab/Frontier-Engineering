#!/usr/bin/env python3
# EVOLVE-BLOCK-START
from __future__ import annotations


def build_charging_policy() -> dict:
    # Aggressive multi-stage charging policy optimized for speed while respecting
    # voltage, temperature, and plating-margin constraints.
    # More stages allow finer control over the charging profile.
    return {
        "currents_c": [
            3.8,   # Stage 1: Very high rate at low SOC (safe plating margin)
            3.5,   # Stage 2: Still aggressive
            3.2,   # Stage 3: Slightly reduced
            2.8,   # Stage 4: Moderate-high
            2.4,   # Stage 5: Moderate
            2.0,   # Stage 6: Reduced as voltage rises
            1.6,   # Stage 7: Lower rate near end
            1.2,   # Stage 8: Gentle finish to target SOC
        ],
        "switch_soc": [
            0.18,  # Switch from stage 1 to 2
            0.30,  # Switch from stage 2 to 3
            0.42,  # Switch from stage 3 to 4
            0.54,  # Switch from stage 4 to 5
            0.64,  # Switch from stage 5 to 6
            0.74,  # Switch from stage 6 to 7
            0.84,  # Switch from stage 7 to 8
        ],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
