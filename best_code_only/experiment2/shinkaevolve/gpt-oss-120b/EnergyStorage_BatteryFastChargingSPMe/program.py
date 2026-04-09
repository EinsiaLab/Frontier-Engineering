#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    return {
        "currents_c": [4.11, 3.61, 3.11, 2.51, 1.61],
        "switch_soc": [0.17, 0.38, 0.58, 0.80],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END