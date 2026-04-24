"""
Official evaluator for MicrowaveAbsorberDesign benchmark.
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

Z0_FREE_SPACE = 377.0
C0 = 2.998e8


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail_result(message: str) -> dict:
    return {"valid": 0, "feasible": 0, "combined_score": 0.0, "message": message}


def validate_submission(submission: dict, config: dict) -> tuple[bool, str]:
    required_keys = [
        "benchmark_id",
        "d_mm",
        "phi_dielectric",
        "phi_magnetic",
        "phi_matrix",
    ]
    for key in required_keys:
        if key not in submission:
            return False, f"Missing required key: '{key}'"
    if submission["benchmark_id"] != config["benchmark_id"]:
        return False, "benchmark_id mismatch"

    d_mm = submission["d_mm"]
    if not isinstance(d_mm, (int, float)) or not math.isfinite(d_mm):
        return False, "d_mm must be finite"
    if not (config["d_mm_min"] <= d_mm <= config["d_mm_max"]):
        return False, "d_mm out of range"

    phis = []
    for key in ["phi_dielectric", "phi_magnetic", "phi_matrix"]:
        val = submission[key]
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            return False, f"{key} must be finite"
        if not (config["phi_min"] <= val <= config["phi_max"]):
            return False, f"{key} out of range"
        phis.append(val)

    if abs(sum(phis) - 1.0) > config["phi_sum_tolerance"]:
        return False, "Volume fractions must sum to 1.0"
    return True, "ok"


def mix_properties(submission: dict, material_db: dict) -> dict:
    phi_d = submission["phi_dielectric"]
    phi_m = submission["phi_magnetic"]
    phi_x = submission["phi_matrix"]
    mat = material_db["matrix"]
    die = material_db["dielectric_filler"]
    mag = material_db["magnetic_filler"]

    eps_real = phi_x * mat["eps_real"] + phi_d * die["eps_real"] + phi_m * mag["eps_real"]
    eps_imag = phi_x * mat["eps_imag"] + phi_d * die["eps_imag"] + phi_m * mag["eps_imag"]
    mu_real = phi_x * mat["mu_real"] + phi_d * die["mu_real"] + phi_m * mag["mu_real"]
    mu_imag = phi_x * mat["mu_imag"] + phi_d * die["mu_imag"] + phi_m * mag["mu_imag"]
    density = phi_x * mat["density"] + phi_d * die["density"] + phi_m * mag["density"]
    cost = phi_x * mat["cost_proxy"] + phi_d * die["cost_proxy"] + phi_m * mag["cost_proxy"]
    return {
        "eps_r": complex(eps_real, -eps_imag),
        "mu_r": complex(mu_real, -mu_imag),
        "density": density,
        "cost": cost,
    }


def compute_rl_curve(eps_r: complex, mu_r: complex, d_mm: float, config: dict):
    freqs_hz = np.linspace(
        config["freq_ghz_min"] * 1e9,
        config["freq_ghz_max"] * 1e9,
        config["num_freq_points"],
    )
    d_m = d_mm * 1e-3
    rl_db = np.zeros(len(freqs_hz))
    for i, freq_hz in enumerate(freqs_hz):
        gamma = 1j * (2.0 * np.pi * freq_hz * d_m / C0) * np.sqrt(mu_r * eps_r)
        z_in = Z0_FREE_SPACE * np.sqrt(mu_r / eps_r) * np.tanh(gamma)
        refl = abs((z_in - Z0_FREE_SPACE) / (z_in + Z0_FREE_SPACE))
        rl_db[i] = 20.0 * np.log10(max(refl, 1e-15))
    return freqs_hz, rl_db


def compute_eab10(freqs_hz: np.ndarray, rl_db: np.ndarray, threshold_db: float = -10.0):
    mask = rl_db <= threshold_db
    if not np.any(mask):
        return 0.0
    max_len = cur_len = end_idx = 0
    for i, flag in enumerate(mask):
        if flag:
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                end_idx = i
        else:
            cur_len = 0
    start_idx = end_idx - max_len + 1
    return (freqs_hz[end_idx] - freqs_hz[start_idx]) / 1e9


def normalize(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def compute_score(rl_min_db, eab10_ghz, d_mm, density, cost, weights, norm):
    return float(
        weights["eab10"] * normalize(eab10_ghz, norm["eab10_ghz"]["min"], norm["eab10_ghz"]["max"])
        + weights["rl_min"]
        * normalize(abs(rl_min_db), norm["abs_rl_min_db"]["min"], norm["abs_rl_min_db"]["max"])
        - weights["thickness"]
        * normalize(d_mm, norm["thickness_mm"]["min"], norm["thickness_mm"]["max"])
        - weights["density"] * normalize(density, norm["density"]["min"], norm["density"]["max"])
        - weights["cost"] * normalize(cost, norm["cost"]["min"], norm["cost"]["max"])
    )


def evaluate_candidate(program_path: Path, task_dir: Path) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(program_path)],
            cwd=str(task_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return fail_result("Candidate program timed out (120s limit)")
    runtime = time.time() - start

    print("=== Candidate stdout ===")
    print(proc.stdout)
    if proc.stderr.strip():
        print("=== Candidate stderr ===")
        print(proc.stderr)

    if proc.returncode != 0:
        return fail_result(f"Candidate exited with code {proc.returncode}")

    submission_path = task_dir / "temp" / "submission.json"
    if not submission_path.exists():
        submission_path = task_dir / "submission.json"
    if not submission_path.exists():
        return fail_result("submission.json not found in temp/ or task root")

    try:
        submission = load_json(submission_path)
    except Exception as exc:
        return fail_result(f"Failed to parse submission.json: {exc}")

    config = load_json(task_dir / "references" / "problem_config.json")
    material_db = load_json(task_dir / "references" / "material_db.json")
    is_valid, msg = validate_submission(submission, config)
    if not is_valid:
        return fail_result(f"Validation failed: {msg}")

    props = mix_properties(submission, material_db)
    freqs_hz, rl_db = compute_rl_curve(props["eps_r"], props["mu_r"], submission["d_mm"], config)
    rl_min_db = float(np.min(rl_db))
    eab10_ghz = compute_eab10(freqs_hz, rl_db, config.get("rl_threshold_db", -10.0))
    combined_score = compute_score(
        rl_min_db,
        eab10_ghz,
        submission["d_mm"],
        props["density"],
        props["cost"],
        config["weights"],
        config["normalization"],
    )
    return {
        "valid": 1,
        "feasible": 1,
        "combined_score": combined_score,
        "rl_min_db": rl_min_db,
        "eab10_ghz": eab10_ghz,
        "thickness_mm": submission["d_mm"],
        "density": props["density"],
        "cost_proxy": props["cost"],
        "runtime_sec": round(runtime, 3),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python verification/evaluator.py <candidate_script>")
        sys.exit(1)

    task_dir = Path(__file__).resolve().parents[1]
    program_path = (task_dir / sys.argv[1]).resolve()
    if not program_path.exists():
        print(f"Error: candidate script not found: {program_path}")
        sys.exit(1)

    result = evaluate_candidate(program_path, task_dir)
    print("\n" + "=" * 50)
    print("  EVALUATION RESULT")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 50)
    if result["valid"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
