#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    return {
        "currents_c": [3.8, 3.2, 2.4, 1.5],
        "switch_soc": [0.25, 0.55, 0.80],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
