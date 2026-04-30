import json
from pathlib import Path


# EVOLVE-BLOCK-START
def generate_scanner_design():
    """
    Generate a simple but valid non-uniform PET scanner design.
    Central rings get slightly thicker crystals than edge rings while
    keeping the total volume comfortably under the budget.
    """
    num_rings = 20
    center = (num_rings - 1) / 2.0
    design = []

    for ring_id in range(num_rings):
        dist = abs(ring_id - center)
        center_weight = max(0.0, 1.0 - dist / center)
        design.append(
            {
                "ring_id": ring_id,
                "R": 400.0,
                "H": round(10.0 + 5.0 * center_weight, 4),
                "W": 4.0,
            }
        )

    return design


# EVOLVE-BLOCK-END


def _output_path() -> Path:
    return Path("solution.json")


if __name__ == "__main__":
    design_data = generate_scanner_design()
    output_path = _output_path()
    output_path.write_text(json.dumps(design_data, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline design successfully generated: {output_path.as_posix()}")
