"""
Minimal initialization for NanoCarbonAbsorberOptimization benchmark.
Mixed-variable: discrete carbon type + continuous content and thickness.
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
        "carbon_type": "CNTs",
        "carbon_content": 0.04,
        "d_mm": 1.5,
    }
    # EVOLVE-BLOCK-END

    output_path = temp_dir / "submission.json"
    output_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    print(f"Submission written to {output_path}")


if __name__ == "__main__":
    main()
