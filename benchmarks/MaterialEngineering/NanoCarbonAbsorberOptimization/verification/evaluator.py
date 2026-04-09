"""
Official evaluator for NanoCarbonAbsorberOptimization benchmark.
Mixed-variable optimization: discrete carbon type + continuous content & thickness.
Nd0.15-BaM/NC composites, 2-18 GHz, PEC backing.

Usage: python verification/evaluator.py scripts/init.py
"""
import json, math, subprocess, sys, time
from pathlib import Path
import numpy as np

Z0 = 377.0
C0 = 2.998e8

def load_json(p):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def fail_result(msg):
    return {"valid": 0, "feasible": 0, "combined_score": 0.0, "message": msg}

def validate_submission(sub, cfg):
    for k in ["benchmark_id", "carbon_type", "carbon_content", "d_mm"]:
        if k not in sub: return False, f"Missing key: '{k}'"
    if sub["benchmark_id"] != cfg["benchmark_id"]:
        return False, "benchmark_id mismatch"

    # Carbon type (discrete variable)
    if sub["carbon_type"] not in cfg["valid_carbon_types"]:
        return False, f"Invalid carbon_type: '{sub['carbon_type']}'. Must be one of {cfg['valid_carbon_types']}"

    # Carbon content (continuous)
    cc = sub["carbon_content"]
    if not isinstance(cc, (int, float)) or not math.isfinite(cc):
        return False, "Invalid carbon_content"
    if not (cfg["carbon_content_min"] <= cc <= cfg["carbon_content_max"]):
        return False, f"carbon_content={cc} out of range [{cfg['carbon_content_min']}, {cfg['carbon_content_max']}]"

    # Thickness (continuous)
    d = sub["d_mm"]
    if not isinstance(d, (int, float)) or not math.isfinite(d):
        return False, "Invalid d_mm"
    if not (cfg["d_mm_min"] <= d <= cfg["d_mm_max"]):
        return False, f"d_mm={d} out of range [{cfg['d_mm_min']}, {cfg['d_mm_max']}]"

    return True, "ok"


def compute_effective_properties(carbon_type, carbon_content, mdb):
    """
    Compute effective electromagnetic properties of the Nd-BaM/NC composite.

    Model: The effective permittivity depends on the carbon type and content.
    Each carbon type has a parametric model fitted from VNA measurements:
        eps_real_eff = base_eps + carbon_eps_real_base * content + carbon_eps_real_slope * content^2
        eps_imag_eff = base_eps_imag + carbon_eps_imag_base * content + carbon_eps_imag_slope * content^2

    Permeability is dominated by Nd-BaM with small offsets from carbon addition.
    """
    base = mdb["base_absorber"]
    carbon = mdb["carbon_materials"][carbon_type]
    cp = carbon["eps_params"]
    mp = carbon["mu_params"]
    cc = carbon_content

    # Effective permittivity: parametric model
    # Linear contribution from carbon loading
    eps_real = base["eps_real"] + cp["eps_real_slope"] * cc
    eps_imag = base["eps_imag"] + cp["eps_imag_slope"] * cc

    # Permeability: primarily from Nd-BaM, reduced slightly by non-magnetic carbon
    mu_real = base["mu_real"] + mp["mu_real_offset"] * (cc / 0.08)
    mu_imag = base["mu_imag"] + mp["mu_imag_offset"] * (cc / 0.08)

    # Effective density: weighted average (carbon replaces some Nd-BaM mass)
    density = (1.0 - cc) * base["density"] + cc * carbon["density"]
    cost = (1.0 - cc) * 1.5 + cc * carbon["cost_proxy"]

    eps_r = complex(eps_real, -eps_imag)
    mu_r = complex(mu_real, -mu_imag)

    return {"eps_r": eps_r, "mu_r": mu_r, "density": density, "cost": cost}


def compute_rl_curve(eps_r, mu_r, d_mm, cfg):
    freqs = np.linspace(cfg["freq_ghz_min"] * 1e9, cfg["freq_ghz_max"] * 1e9, cfg["num_freq_points"])
    d_m = d_mm * 1e-3
    rl = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        g = 1j * (2 * np.pi * f * d_m / C0) * np.sqrt(mu_r * eps_r)
        zi = Z0 * np.sqrt(mu_r / eps_r) * np.tanh(g)
        r = abs((zi - Z0) / (zi + Z0))
        rl[i] = 20.0 * np.log10(max(r, 1e-15))
    return freqs, rl


def compute_eab10(freqs, rl, thr=-10.0):
    mask = rl <= thr
    if not np.any(mask): return 0.0
    ml = cl = ei = 0
    for i, f in enumerate(mask):
        if f:
            cl += 1
            if cl > ml: ml = cl; ei = i
        else:
            cl = 0
    if ml == 0: return 0.0
    return (freqs[ei] - freqs[ei - ml + 1]) / 1e9


def norm(v, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def compute_score(rl_min, eab, d, dens, cost, w, n):
    return float(
        w["eab10"] * norm(eab, n["eab10_ghz"]["min"], n["eab10_ghz"]["max"])
        + w["rl_min"] * norm(abs(rl_min), n["abs_rl_min_db"]["min"], n["abs_rl_min_db"]["max"])
        - w["thickness"] * norm(d, n["thickness_mm"]["min"], n["thickness_mm"]["max"])
        - w["density"] * norm(dens, n["density"]["min"], n["density"]["max"])
        - w["cost"] * norm(cost, n["cost"]["min"], n["cost"]["max"])
    )


def evaluate_candidate(prog, task_dir):
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, str(prog)], cwd=str(task_dir),
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return fail_result("Timeout (120s)")
    runtime = time.time() - t0
    print("=== Candidate stdout ==="); print(proc.stdout)
    if proc.stderr.strip(): print("=== stderr ==="); print(proc.stderr)
    if proc.returncode != 0: return fail_result(f"Exit code {proc.returncode}")

    sp = task_dir / "temp" / "submission.json"
    if not sp.exists(): sp = task_dir / "submission.json"
    if not sp.exists(): return fail_result("submission.json not found")
    try: sub = load_json(sp)
    except Exception as e: return fail_result(f"Parse error: {e}")

    cfg = load_json(task_dir / "references" / "problem_config.json")
    mdb = load_json(task_dir / "references" / "material_db.json")
    ok, msg = validate_submission(sub, cfg)
    if not ok: return fail_result(f"Validation: {msg}")

    props = compute_effective_properties(sub["carbon_type"], sub["carbon_content"], mdb)
    freqs, rl = compute_rl_curve(props["eps_r"], props["mu_r"], sub["d_mm"], cfg)
    rl_min = float(np.min(rl))
    eab = compute_eab10(freqs, rl, cfg.get("rl_threshold_db", -10.0))

    base = {"carbon_type": sub["carbon_type"], "carbon_content": sub["carbon_content"],
            "rl_min_db": rl_min, "eab10_ghz": eab, "thickness_mm": sub["d_mm"],
            "density": props["density"], "cost_proxy": props["cost"], "runtime_sec": round(runtime, 3)}

    min_eab = cfg.get("min_eab_ghz", 0.0)
    if eab < min_eab:
        return {**base, "valid": 1, "feasible": 0, "combined_score": 0.0,
                "message": f"EAB={eab:.2f} GHz < min required {min_eab} GHz"}

    score = compute_score(rl_min, eab, sub["d_mm"], props["density"], props["cost"],
                          cfg["weights"], cfg["normalization"])
    return {**base, "valid": 1, "feasible": 1, "combined_score": score}


def main():
    if len(sys.argv) < 2: print("Usage: python verification/evaluator.py <script>"); sys.exit(1)
    task_dir = Path(__file__).resolve().parents[1]
    prog = (task_dir / sys.argv[1]).resolve()
    if not prog.exists(): print(f"Not found: {prog}"); sys.exit(1)
    result = evaluate_candidate(prog, task_dir)
    print("\n" + "=" * 50 + "\n  EVALUATION RESULT\n" + "=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 50)
    if result["valid"] == 0: sys.exit(1)

if __name__ == "__main__": main()
