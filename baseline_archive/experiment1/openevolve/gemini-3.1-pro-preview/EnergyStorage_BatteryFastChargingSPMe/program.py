#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    return {
        "currents_c": [3.62, 3.38, 2.78, 1.58],
        "switch_soc": [0.33, 0.65, 0.86],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
