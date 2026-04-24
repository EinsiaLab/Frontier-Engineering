"""
Minimal initialization script for MicrowaveAbsorberDesign benchmark.
Generates a valid submission with a simple design.
"""
import json
from pathlib import Path


def main():
    task_dir = Path(__file__).resolve().parents[1]
    temp_dir = task_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    config = json.loads((task_dir / "references" / "problem_config.json").read_text())

    # EVOLVE-BLOCK-START
    submission = {
        "benchmark_id": config["benchmark_id"],
        "d_mm": 2.0,
        "phi_dielectric": 0.45,
        "phi_magnetic": 0.45,
        "phi_matrix": 0.10,
    }
    # EVOLVE-BLOCK-END

    output_path = temp_dir / "submission.json"
    output_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    print(f"Submission written to {output_path}")


if __name__ == "__main__":
    main()
