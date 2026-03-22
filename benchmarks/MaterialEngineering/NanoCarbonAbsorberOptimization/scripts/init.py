"""
Minimal initialization for NanoCarbonAbsorberOptimization benchmark.
Mixed-variable: discrete carbon type + continuous content and thickness.
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
    # Optimize a Nd0.15-BaM/nano-carbon composite absorber for 2-18 GHz.
    # Variables:
    #   carbon_type: "CNTs" or "GO" or "OLC" (discrete choice)
    #   carbon_content: mass fraction of nano-carbon [0.01, 0.10]
    #   d_mm: absorber thickness in mm [1.5, 5.0]
    # Hard constraint: EAB >= 3.0 GHz (RL <= -10 dB continuous bandwidth)
    # Goal: maximize combined_score (wide EAB, deep RL, thin, light, cheap)

    submission = {
        "benchmark_id": config["benchmark_id"],
        "carbon_type": "CNTs",
        "carbon_content": 0.04,
        "d_mm": 1.5
    }
    # EVOLVE-BLOCK-END

    output_path = temp_dir / "submission.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission written to {output_path}")


if __name__ == "__main__":
    main()
