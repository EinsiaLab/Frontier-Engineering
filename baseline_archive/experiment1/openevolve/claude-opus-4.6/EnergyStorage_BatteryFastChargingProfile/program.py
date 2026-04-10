#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    """Aggressive multi-stage constant-current fast-charge profile.
    
    Push higher currents longer to minimize charge time while staying
    within thermal and voltage safety margins. Previous profile was
    too conservative (temp=26.8C, voltage=4.04V) with plenty of headroom.
    """
    return {
        "currents_c": [6.0, 5.8, 5.2, 4.3, 3.5, 2.8],
        "switch_soc": [0.60, 0.68, 0.73, 0.77, 0.79],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
