"""
Official evaluator for MicrowaveAbsorberDesign benchmark.
Evaluates a single-layer microwave absorber design in the X-band (8-12 GHz)
using transmission line theory with PEC backing.

Usage:
    python verification/evaluator.py scripts/init.py
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# Physical constants
# ============================================================
Z0_FREE_SPACE = 377.0       # Impedance of free space (Ohm)
C0 = 2.998e8                # Speed of light in vacuum (m/s)


# ============================================================
# File I/O helpers
# ============================================================
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fail_result(message: str) -> dict:
    """Return a standardized failure result."""
    return {
        "valid": 0,
        "feasible": 0,
        "combined_score": 0.0,
        "message": message,
    }


# ============================================================
# Input validation
# ============================================================
def validate_submission(submission: dict, config: dict) -> tuple:
    """
    Validate that the submission JSON conforms to expected format.
    Returns (is_valid: bool, message: str).
    """
    required_keys = [
        "benchmark_id", "d_mm",
        "phi_dielectric", "phi_magnetic", "phi_matrix",
    ]
    for key in required_keys:
        if key not in submission:
            return False, f"Missing required key: '{key}'"

    if submission["benchmark_id"] != config["benchmark_id"]:
        return False, (
            f"benchmark_id mismatch: expected '{config['benchmark_id']}', "
            f"got '{submission['benchmark_id']}'"
        )

    # Thickness check
    d_mm = submission["d_mm"]
    if not isinstance(d_mm, (int, float)) or not math.isfinite(d_mm):
        return False, f"d_mm must be a finite number, got {d_mm}"
    if not (config["d_mm_min"] <= d_mm <= config["d_mm_max"]):
        return False, (
            f"d_mm={d_mm} out of range [{config['d_mm_min']}, {config['d_mm_max']}]"
        )

    # Volume fraction checks
    phi_keys = ["phi_dielectric", "phi_magnetic", "phi_matrix"]
    phis = []
    for key in phi_keys:
        val = submission[key]
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            return False, f"{key} must be a finite number, got {val}"
        if val < config["phi_min"] or val > config["phi_max"]:
            return False, f"{key}={val} out of range [{config['phi_min']}, {config['phi_max']}]"
        phis.append(val)

    phi_sum = sum(phis)
    if abs(phi_sum - 1.0) > config["phi_sum_tolerance"]:
        return False, (
            f"Volume fractions must sum to 1.0 (got {phi_sum:.10f}, "
            f"tolerance={config['phi_sum_tolerance']})"
        )

    return True, "ok"


# ============================================================
# Material property mixing (linear rule of mixtures)
# ============================================================
def mix_properties(submission: dict, material_db: dict) -> dict:
    """
    Compute effective complex permittivity, permeability, density, and cost
    using a linear volume-fraction mixing rule.

    Convention: eps_r = eps_real - j * eps_imag  (negative imaginary part)
                mu_r  = mu_real  - j * mu_imag
    """
    phi_d = submission["phi_dielectric"]
    phi_m = submission["phi_magnetic"]
    phi_x = submission["phi_matrix"]

    mat = material_db["matrix"]
    die = material_db["dielectric_filler"]
    mag = material_db["magnetic_filler"]

    eps_real = phi_x * mat["eps_real"] + phi_d * die["eps_real"] + phi_m * mag["eps_real"]
    eps_imag = phi_x * mat["eps_imag"] + phi_d * die["eps_imag"] + phi_m * mag["eps_imag"]
    mu_real  = phi_x * mat["mu_real"]  + phi_d * die["mu_real"]  + phi_m * mag["mu_real"]
    mu_imag  = phi_x * mat["mu_imag"]  + phi_d * die["mu_imag"]  + phi_m * mag["mu_imag"]

    density = phi_x * mat["density"] + phi_d * die["density"] + phi_m * mag["density"]
    cost    = phi_x * mat["cost_proxy"] + phi_d * die["cost_proxy"] + phi_m * mag["cost_proxy"]

    # Complex values with negative-imaginary-part convention
    eps_r = complex(eps_real, -eps_imag)
    mu_r  = complex(mu_real,  -mu_imag)

    return {
        "eps_r": eps_r,
        "mu_r": mu_r,
        "density": density,
        "cost": cost,
    }


# ============================================================
# Reflection loss computation (transmission line theory)
# ============================================================
def compute_rl_curve(eps_r: complex, mu_r: complex,
                     d_mm: float, config: dict) -> tuple:
    """
    Compute the reflection loss (RL) curve for a single-layer absorber
    backed by a perfect electrical conductor (PEC).

    Physical model:
        Z_in = Z0 * sqrt(mu_r / eps_r) * tanh(j * 2*pi*f*d/c * sqrt(mu_r * eps_r))
        RL(f) = 20 * log10(|Z_in - Z0| / |Z_in + Z0|)
    """
    fmin_hz = config["freq_ghz_min"] * 1e9
    fmax_hz = config["freq_ghz_max"] * 1e9
    npts = config["num_freq_points"]

    freqs_hz = np.linspace(fmin_hz, fmax_hz, npts)
    d_m = d_mm * 1e-3  # Convert mm to meters

    rl_db = np.zeros(npts)
    for i, f in enumerate(freqs_hz):
        gamma = 1j * (2.0 * np.pi * f * d_m / C0) * np.sqrt(mu_r * eps_r)
        z_in = Z0_FREE_SPACE * np.sqrt(mu_r / eps_r) * np.tanh(gamma)
        refl = abs((z_in - Z0_FREE_SPACE) / (z_in + Z0_FREE_SPACE))
        rl_db[i] = 20.0 * np.log10(max(refl, 1e-15))

    return freqs_hz, rl_db


# ============================================================
# Effective absorption bandwidth (maximum continuous span)
# ============================================================
def compute_eab10(freqs_hz: np.ndarray, rl_db: np.ndarray,
                  threshold_db: float = -10.0) -> float:
    """
    Compute the maximum continuous bandwidth (in GHz) where RL <= threshold_db.
    """
    mask = rl_db <= threshold_db
    if not np.any(mask):
        return 0.0

    max_len = 0
    cur_len = 0
    end_idx = 0
    for i, flag in enumerate(mask):
        if flag:
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                end_idx = i
        else:
            cur_len = 0

    if max_len == 0:
        return 0.0

    start_idx = end_idx - max_len + 1
    bw_hz = freqs_hz[end_idx] - freqs_hz[start_idx]
    return bw_hz / 1e9


# ============================================================
# Min-max normalization helper
# ============================================================
def normalize(value: float, vmin: float, vmax: float) -> float:
    """Normalize a value to [0, 1] range using min-max scaling. Clamps to bounds."""
    if vmax <= vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


# ============================================================
# Combined scoring (with normalization)
# ============================================================
def compute_score(rl_min_db: float, eab10_ghz: float,
                  d_mm: float, density: float, cost: float,
                  weights: dict, norm: dict) -> float:
    """
    Compute the combined benchmark score with min-max normalization.

    Each metric is first normalized to [0, 1] using predefined ranges,
    then weighted and combined:

        combined_score = w_eab10 * norm(EAB_10)
                       + w_rl_min * norm(|RL_min|)
                       - w_thickness * norm(d_mm)
                       - w_density * norm(density)
                       - w_cost * norm(cost)

    Higher is better.
    """
    n_eab  = normalize(eab10_ghz,   norm["eab10_ghz"]["min"],     norm["eab10_ghz"]["max"])
    n_rl   = normalize(abs(rl_min_db), norm["abs_rl_min_db"]["min"], norm["abs_rl_min_db"]["max"])
    n_d    = normalize(d_mm,         norm["thickness_mm"]["min"],   norm["thickness_mm"]["max"])
    n_rho  = normalize(density,      norm["density"]["min"],        norm["density"]["max"])
    n_cost = normalize(cost,         norm["cost"]["min"],           norm["cost"]["max"])

    score = (
        weights["eab10"]     * n_eab
        + weights["rl_min"]  * n_rl
        - weights["thickness"] * n_d
        - weights["density"]   * n_rho
        - weights["cost"]      * n_cost
    )
    return float(score)


# ============================================================
# Main evaluation pipeline
# ============================================================
def evaluate_candidate(program_path: Path, task_dir: Path) -> dict:
    """
    Run a candidate program and evaluate its submission.
    """
    # ------ Step 1: Run candidate program ------
    t0 = time.time()
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
    runtime = time.time() - t0

    print("=== Candidate stdout ===")
    print(proc.stdout)
    if proc.stderr.strip():
        print("=== Candidate stderr ===")
        print(proc.stderr)

    if proc.returncode != 0:
        return fail_result(f"Candidate exited with code {proc.returncode}")

    # ------ Step 2: Load submission ------
    submission_path = task_dir / "temp" / "submission.json"
    if not submission_path.exists():
        submission_path = task_dir / "submission.json"
    if not submission_path.exists():
        return fail_result("submission.json not found in temp/ or task root")

    try:
        submission = load_json(submission_path)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return fail_result(f"Failed to parse submission.json: {e}")

    # ------ Step 3: Load config & validate ------
    config = load_json(task_dir / "references" / "problem_config.json")
    material_db = load_json(task_dir / "references" / "material_db.json")

    is_valid, msg = validate_submission(submission, config)
    if not is_valid:
        return fail_result(f"Validation failed: {msg}")

    # ------ Step 4: Compute material properties ------
    props = mix_properties(submission, material_db)

    # ------ Step 5: Compute RL curve ------
    freqs_hz, rl_db = compute_rl_curve(
        props["eps_r"], props["mu_r"], submission["d_mm"], config
    )

    rl_min_db = float(np.min(rl_db))
    threshold = config.get("rl_threshold_db", -10.0)
    eab10_ghz = compute_eab10(freqs_hz, rl_db, threshold)

    # ------ Step 6: Score (with normalization) ------
    combined_score = compute_score(
        rl_min_db, eab10_ghz,
        submission["d_mm"], props["density"], props["cost"],
        config["weights"], config["normalization"],
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


# ============================================================
# CLI entry point
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python verification/evaluator.py <candidate_script>")
        print("Example: python verification/evaluator.py scripts/init.py")
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
