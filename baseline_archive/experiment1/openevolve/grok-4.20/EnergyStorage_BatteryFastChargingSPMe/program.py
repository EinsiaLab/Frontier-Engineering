#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    # 4-stage policy tuned for best observed balance.
    # Keeps max_voltage_v just above 4.20 (still high voltage_score)
    # while shortening charge time relative to more conservative schedules.
    # Plating and thermal margins stay large; aging remains negligible.
    base_currents = [3.65, 3.05, 2.25, 1.35]
    soc_points = [0.29, 0.59, 0.84]
    currents_c = [c * 0.995 for c in base_currents]
    switch_soc = soc_points
    return {
        "currents_c": currents_c,
        "switch_soc": switch_soc,
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
