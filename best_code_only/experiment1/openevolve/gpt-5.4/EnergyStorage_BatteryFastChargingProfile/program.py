#!/usr/bin/env python3
# EVOLVE-BLOCK-START

def build_charging_profile() -> dict:
    return {
        # Slightly more aggressive mid-SOC charging while preserving
        # a conservative final taper near the high-SOC voltage knee.
        "currents_c": [6.0, 5.0, 3.0, 1.1],
        "switch_soc": [0.49, 0.71, 0.76],
    }


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
