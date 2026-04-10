#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    return {
        "currents_c": [3.7, 3.6, 2.9, 2.0],
        "switch_soc": [0.22, 0.55, 0.80],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
