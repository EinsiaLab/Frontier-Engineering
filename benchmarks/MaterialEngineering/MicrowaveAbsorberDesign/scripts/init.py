"""
Minimal initialization script for MicrowaveAbsorberDesign benchmark.
Generates a valid submission with a simple design.
This is the target file for agent evolution.
"""
import json
from pathlib import Path


def main():
    task_dir = Path(__file__).resolve().parents[1]
    temp_dir = task_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    config_path = task_dir / "references" / "problem_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # EVOLVE-BLOCK-START
    # Design a single-layer microwave absorber for X-band (8-12 GHz).
    # Variables:
    #   d_mm: absorber thickness in millimeters [1.0, 5.0]
    #   phi_dielectric: volume fraction of dielectric filler [0, 1]
    #   phi_magnetic: volume fraction of magnetic filler [0, 1]
    #   phi_matrix: volume fraction of matrix [0, 1]
    # Constraint: phi_dielectric + phi_magnetic + phi_matrix == 1.0
    # Goal: maximize combined_score (wider bandwidth, deeper RL, thinner, lighter, cheaper)

    submission = {
        "benchmark_id": config["benchmark_id"],
        "d_mm": 2.0,
        "phi_dielectric": 0.45,
        "phi_magnetic": 0.45,
        "phi_matrix": 0.10
    }
    # EVOLVE-BLOCK-END

    output_path = temp_dir / "submission.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    print(f"Submission written to {output_path}")


if __name__ == "__main__":
    main()
