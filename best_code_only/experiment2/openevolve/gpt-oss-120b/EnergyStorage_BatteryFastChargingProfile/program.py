#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations

TARGET_SOC: float = 0.80




def build_charging_profile() -> dict:
    """Fast multi‑stage constant‑current charging profile."""
    # Optimized multi‑stage charging profile.
    return {
        "currents_c": [6.0, 5.0, 4.0, 3.4],
        # Slightly increase the early SOC break‑points to extend the
        # high‑current stages (mirrors the highest‑scoring configuration).
        "switch_soc": [0.38, 0.62, 0.78],
    }


def _validate_profile(profile: dict) -> None:
    """Validate the charging profile structure."""
    if not isinstance(profile, dict):
        raise TypeError("profile must be a dict")
    if "currents_c" not in profile or "switch_soc" not in profile:
        raise KeyError("profile missing required keys")
    if not all(isinstance(v, (int, float)) for v in profile["currents_c"]):
        raise TypeError("currents_c values must be numbers")
    if not all(isinstance(v, (int, float)) for v in profile["switch_soc"]):
        raise TypeError("switch_soc values must be numbers")

def main() -> None:
    profile = build_charging_profile()
    _validate_profile(profile)
    print(profile)


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
