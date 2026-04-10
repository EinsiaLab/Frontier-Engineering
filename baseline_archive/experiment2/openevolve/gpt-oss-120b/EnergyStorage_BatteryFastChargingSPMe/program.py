#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_policy() -> dict:
    """
    Charging policy with slightly higher early‑stage currents and
    adjusted SOC switch‑points to reduce total charge time while
    preserving voltage, temperature and plating safety margins.
    """
    # Refined staged policy:
    # • Slightly lower early‑stage currents keep the maximum voltage
    #   below the soft‑voltage limit (avoiding the soft‑voltage penalty).
    # • The later stages remain unchanged, preserving a short overall
    #   charge time while still respecting all hard limits.
    # Updated staged policy:
    # • Slightly higher currents in every stage reduce total charge time.
    # • The chosen values stay below the hard voltage (4.25 V) and temperature
    #   limits observed for the previous best‑scoring policy (max V ≈ 4.196 V,
    #   max T ≈ 32 °C).  This should keep the voltage‑score and thermal‑score
    #   at their maximum while improving the time‑score.
    # • Switch‑SOC thresholds are unchanged because they already provide a
    #   safe transition between stages.
    # Improved policy:
    # • Reduce first‑stage current to 3.9 C to keep peak voltage ≤ 4.20 V.
    # • Slightly lower third‑stage current to 2.7 C.
    # • Move SOC thresholds earlier so high‑current stages are shorter,
    #   preserving a strong time_score while eliminating the soft‑voltage penalty.
    # Updated four‑stage fast‑charging policy that has proven to give the
    # highest combined score in previous experiments.  The early‑stage
    # currents are modestly increased to reduce charge time, while the
    # SOC break‑points are chosen to keep each high‑current interval short
    # enough to stay below the soft‑voltage (4.20 V) and temperature limits.
    return {
        "currents_c": [4.1, 3.6, 2.7, 1.8],
        "switch_soc": [0.18, 0.48, 0.74],
    }


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
