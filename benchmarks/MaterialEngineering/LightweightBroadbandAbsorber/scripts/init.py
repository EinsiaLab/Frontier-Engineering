"""
Minimal initialization for LightweightBroadbandAbsorber benchmark.
<<<<<<< Updated upstream
Based on CNTs@Nd0.15-BaM/PE composite system.
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    # Design a lightweight broadband absorber using CNTs@Nd-BaM/PE system (8.2-18 GHz).
    # Variables:
    #   d_mm: absorber thickness in mm [1.0, 5.0]
    #   phi_magnetic_absorber: Nd0.15-BaM volume fraction [0, 1]
    #   phi_conductive_filler: CNTs volume fraction [0, 1]
    #   phi_lightweight_magnetic: hollow Nd-BaM microspheres [0, 1]
    #   phi_matrix: PE matrix [0, 1]
    # Constraint: all phi sum to 1.0
    # Hard constraint: EAB >= 4.0 GHz (RL <= -10 dB continuous bandwidth)
    # Goal: maximize combined_score (wide EAB, deep RL, thin, LIGHT, cheap)

    submission = {
        "benchmark_id": config["benchmark_id"],
        "d_mm": 1.9,
        "phi_magnetic_absorber": 0.25,
        "phi_conductive_filler": 0.10,
        "phi_lightweight_magnetic": 0.05,
        "phi_matrix": 0.60
=======
    # Design a lightweight broadband microwave absorber for 2-18 GHz.
    # Variables:
    #   d_mm: absorber thickness in mm [1.0, 6.0]
    #   phi_dielectric: volume fraction of dielectric filler [0, 1]
    #   phi_magnetic: volume fraction of magnetic filler [0, 1]
    #   phi_lightweight_magnetic: volume fraction of lightweight magnetic filler [0, 1]
    #   phi_matrix: volume fraction of matrix [0, 1]
    # Constraint: all phi sum to 1.0
    # Hard constraint: EAB >= 4.0 GHz (otherwise infeasible)
    # Goal: maximize combined_score (wide bandwidth, deep RL, thin, LIGHT, cheap)

    submission = {
        "benchmark_id": config["benchmark_id"],
        "d_mm": 1.5,
        "phi_dielectric": 0.40,
        "phi_magnetic": 0.40,
        "phi_lightweight_magnetic": 0.05,
        "phi_matrix": 0.15
>>>>>>> Stashed changes
    }
    # EVOLVE-BLOCK-END

    output_path = temp_dir / "submission.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission written to {output_path}")


if __name__ == "__main__":
    main()
