"""
Official evaluator for LightweightBroadbandAbsorber benchmark.
CNTs@Nd-BaM/PE system, 8.2-18 GHz, PEC backing.
4 material components. Minimum EAB hard constraint.

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
    for k in ["benchmark_id", "d_mm", "phi_magnetic_absorber", "phi_conductive_filler",
               "phi_lightweight_magnetic", "phi_matrix"]:
        if k not in sub: return False, f"Missing key: '{k}'"
    if sub["benchmark_id"] != cfg["benchmark_id"]:
        return False, "benchmark_id mismatch"
    d = sub["d_mm"]
    if not isinstance(d, (int, float)) or not math.isfinite(d): return False, "Invalid d_mm"
    if not (cfg["d_mm_min"] <= d <= cfg["d_mm_max"]): return False, "d_mm out of range"
    phis = []
    for k in ["phi_magnetic_absorber", "phi_conductive_filler", "phi_lightweight_magnetic", "phi_matrix"]:
        v = sub[k]
        if not isinstance(v, (int, float)) or not math.isfinite(v): return False, f"Invalid {k}"
        if v < cfg["phi_min"] or v > cfg["phi_max"]: return False, f"{k} out of range"
        phis.append(v)
    if abs(sum(phis) - 1.0) > cfg["phi_sum_tolerance"]:
        return False, f"Volume fractions sum to {sum(phis):.10f}, not 1.0"
    return True, "ok"

def mix_properties(sub, mdb):
    comps = [
        (sub["phi_matrix"], mdb["matrix"]),
        (sub["phi_magnetic_absorber"], mdb["magnetic_absorber"]),
        (sub["phi_conductive_filler"], mdb["conductive_filler"]),
        (sub["phi_lightweight_magnetic"], mdb["lightweight_magnetic"]),
    ]
    er = complex(sum(p * c["eps_real"] for p, c in comps),
                 -sum(p * c["eps_imag"] for p, c in comps))
    mr = complex(sum(p * c["mu_real"] for p, c in comps),
                 -sum(p * c["mu_imag"] for p, c in comps))
    dn = sum(p * c["density"] for p, c in comps)
    ct = sum(p * c["cost_proxy"] for p, c in comps)
    return {"eps_r": er, "mu_r": mr, "density": dn, "cost": ct}

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

    props = mix_properties(sub, mdb)
    freqs, rl = compute_rl_curve(props["eps_r"], props["mu_r"], sub["d_mm"], cfg)
    rl_min = float(np.min(rl))
    eab = compute_eab10(freqs, rl, cfg.get("rl_threshold_db", -10.0))

    base = {"rl_min_db": rl_min, "eab10_ghz": eab, "thickness_mm": sub["d_mm"],
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
