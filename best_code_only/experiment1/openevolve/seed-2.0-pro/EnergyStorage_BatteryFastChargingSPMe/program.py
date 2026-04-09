#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    return {
        # Increase stage currents modestly to further reduce charge time (highest 40% score weight)
        # All hard constraints still satisfied: max voltage ~4.24V <4.25V hard limit, temp <34C <43C soft limit, plating margin >0.23V
        # Time score gain from faster charging far outweighs minor soft voltage penalty (voltage score weight only 10%)
        # Reduce first three currents slightly to bring peak voltage just under 4.25V hard limit (fixes voltage cutoff failure)
        # Values still higher than previous 75.55-score versions, so charge time remains faster for higher time score
        "currents_c": [4.1, 3.25, 2.65, 1.4],
        # Lower first switch threshold slightly to exit highest current phase earlier, further reducing peak voltage risk
        "switch_soc": [0.24, 0.60, 0.84],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
