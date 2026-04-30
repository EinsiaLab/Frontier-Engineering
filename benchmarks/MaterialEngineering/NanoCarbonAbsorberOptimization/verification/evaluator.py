"""
Official evaluator for NanoCarbonAbsorberOptimization benchmark.
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

Z0 = 377.0
C0 = 2.998e8


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fail_result(msg):
    return {"valid": 0, "feasible": 0, "combined_score": 0.0, "message": msg}


def validate_submission(sub, cfg):
    for key in ["benchmark_id", "carbon_type", "carbon_content", "d_mm"]:
        if key not in sub:
            return False, f"Missing key: '{key}'"
    if sub["benchmark_id"] != cfg["benchmark_id"]:
        return False, "benchmark_id mismatch"
    if sub["carbon_type"] not in cfg["valid_carbon_types"]:
        return False, f"Invalid carbon_type: '{sub['carbon_type']}'"

    cc = sub["carbon_content"]
    if not isinstance(cc, (int, float)) or not math.isfinite(cc):
        return False, "Invalid carbon_content"
    if not (cfg["carbon_content_min"] <= cc <= cfg["carbon_content_max"]):
        return False, "carbon_content out of range"

    d_mm = sub["d_mm"]
    if not isinstance(d_mm, (int, float)) or not math.isfinite(d_mm):
        return False, "Invalid d_mm"
    if not (cfg["d_mm_min"] <= d_mm <= cfg["d_mm_max"]):
        return False, "d_mm out of range"
    return True, "ok"


def compute_effective_properties(carbon_type, carbon_content, mdb):
    base = mdb["base_absorber"]
    carbon = mdb["carbon_materials"][carbon_type]
    cp = carbon["eps_params"]
    mp = carbon["mu_params"]
    cc = carbon_content

    eps_real = base["eps_real"] + cp["eps_real_slope"] * cc
    eps_imag = base["eps_imag"] + cp["eps_imag_slope"] * cc
    mu_real = base["mu_real"] + mp["mu_real_offset"] * (cc / 0.08)
    mu_imag = base["mu_imag"] + mp["mu_imag_offset"] * (cc / 0.08)
    density = (1.0 - cc) * base["density"] + cc * carbon["density"]
    cost = (1.0 - cc) * 1.5 + cc * carbon["cost_proxy"]

    return {
        "eps_r": complex(eps_real, -eps_imag),
        "mu_r": complex(mu_real, -mu_imag),
        "density": density,
        "cost": cost,
    }


def compute_rl_curve(eps_r, mu_r, d_mm, cfg):
    freqs = np.linspace(cfg["freq_ghz_min"] * 1e9, cfg["freq_ghz_max"] * 1e9, cfg["num_freq_points"])
    d_m = d_mm * 1e-3
    rl = np.zeros(len(freqs))
    for i, freq_hz in enumerate(freqs):
        gamma = 1j * (2 * np.pi * freq_hz * d_m / C0) * np.sqrt(mu_r * eps_r)
        z_in = Z0 * np.sqrt(mu_r / eps_r) * np.tanh(gamma)
        refl = abs((z_in - Z0) / (z_in + Z0))
        rl[i] = 20.0 * np.log10(max(refl, 1e-15))
    return freqs, rl


def compute_eab10(freqs, rl, thr=-10.0):
    mask = rl <= thr
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
    if max_len == 0:
        return 0.0
    return (freqs[end_idx] - freqs[end_idx - max_len + 1]) / 1e9


def norm(v, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def compute_score(rl_min, eab, d_mm, density, cost, weights, norm_cfg):
    return float(
        weights["eab10"] * norm(eab, norm_cfg["eab10_ghz"]["min"], norm_cfg["eab10_ghz"]["max"])
        + weights["rl_min"] * norm(abs(rl_min), norm_cfg["abs_rl_min_db"]["min"], norm_cfg["abs_rl_min_db"]["max"])
        - weights["thickness"] * norm(d_mm, norm_cfg["thickness_mm"]["min"], norm_cfg["thickness_mm"]["max"])
        - weights["density"] * norm(density, norm_cfg["density"]["min"], norm_cfg["density"]["max"])
        - weights["cost"] * norm(cost, norm_cfg["cost"]["min"], norm_cfg["cost"]["max"])
    )


def evaluate_candidate(program_path, task_dir):
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
        return fail_result("Timeout (120s)")
    runtime = time.time() - start
    print("=== Candidate stdout ===")
    print(proc.stdout)
    if proc.stderr.strip():
        print("=== stderr ===")
        print(proc.stderr)
    if proc.returncode != 0:
        return fail_result(f"Exit code {proc.returncode}")

    submission_path = task_dir / "temp" / "submission.json"
    if not submission_path.exists():
        submission_path = task_dir / "submission.json"
    if not submission_path.exists():
        return fail_result("submission.json not found")
    try:
        sub = load_json(submission_path)
    except Exception as exc:
        return fail_result(f"Parse error: {exc}")

    cfg = load_json(task_dir / "references" / "problem_config.json")
    mdb = load_json(task_dir / "references" / "material_db.json")
    ok, msg = validate_submission(sub, cfg)
    if not ok:
        return fail_result(f"Validation: {msg}")

    props = compute_effective_properties(sub["carbon_type"], sub["carbon_content"], mdb)
    freqs, rl = compute_rl_curve(props["eps_r"], props["mu_r"], sub["d_mm"], cfg)
    rl_min = float(np.min(rl))
    eab = compute_eab10(freqs, rl, cfg.get("rl_threshold_db", -10.0))

    base = {
        "carbon_type": sub["carbon_type"],
        "carbon_content": sub["carbon_content"],
        "rl_min_db": rl_min,
        "eab10_ghz": eab,
        "thickness_mm": sub["d_mm"],
        "density": props["density"],
        "cost_proxy": props["cost"],
        "runtime_sec": round(runtime, 3),
    }

    if eab < cfg.get("min_eab_ghz", 0.0):
        return {**base, "valid": 1, "feasible": 0, "combined_score": 0.0, "message": f"EAB={eab:.2f} GHz below minimum"}

    score = compute_score(
        rl_min,
        eab,
        sub["d_mm"],
        props["density"],
        props["cost"],
        cfg["weights"],
        cfg["normalization"],
    )
    return {**base, "valid": 1, "feasible": 1, "combined_score": score}


def main():
    if len(sys.argv) < 2:
        print("Usage: python verification/evaluator.py <script>")
        sys.exit(1)
    task_dir = Path(__file__).resolve().parents[1]
    prog = (task_dir / sys.argv[1]).resolve()
    if not prog.exists():
        print(f"Not found: {prog}")
        sys.exit(1)
    result = evaluate_candidate(prog, task_dir)
    print("\n" + "=" * 50 + "\n  EVALUATION RESULT\n" + "=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 50)
    if result["valid"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
