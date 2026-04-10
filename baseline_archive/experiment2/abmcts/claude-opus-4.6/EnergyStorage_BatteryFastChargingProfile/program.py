#!/usr/bin/env python3
# EVOLVE-BLOCK-START
from __future__ import annotations


def build_charging_profile() -> dict:
    """Aggressively optimized multi-stage constant-current fast-charge profile.
    
    Strategy: Push currents as high as possible while staying within hard limits.
    
    From parent run analysis:
    - charge_time_s=647, max_temp=28.5C, max_voltage=4.07V
    - Still massive thermal headroom (47C limit vs 28.5C)
    - Still significant voltage headroom (4.25V limit vs 4.07V)
    - time_score=0.945 is good but can be improved further
    - degradation_score=0.9999 is excellent
    
    Push even harder: use max current (6.0C) for longer, and keep later stages
    higher since we have so much headroom on both temperature and voltage.
    """
    return {
        "currents_c": [6.0, 6.0, 5.5, 4.5, 3.5, 2.0],
        "switch_soc": [0.35, 0.50, 0.62, 0.72, 0.77],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
