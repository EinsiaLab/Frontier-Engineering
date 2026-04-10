#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    # Optimized 5-stage policy for better time and aging balance
    # Slightly higher currents in middle and final stages to reduce charging time
    # Fine-tuned switching points to better utilize voltage headroom
    # Adjusted to reduce aging stress while maintaining safety margins
    return {
        "currents_c": [3.7, 3.2, 2.7, 2.1, 1.4],
        "switch_soc": [0.22, 0.48, 0.73, 0.84],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
