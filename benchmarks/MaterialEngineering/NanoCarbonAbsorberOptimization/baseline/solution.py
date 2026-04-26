"""
Baseline for NanoCarbonAbsorberOptimization.
Searches across all three carbon types with random content and thickness.
"""
import json, random
from pathlib import Path
import numpy as np

Z0, C0 = 377.0, 2.998e8

def norm(v, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

def get_props(ctype, cc, mdb):
    base = mdb["base_absorber"]
    carbon = mdb["carbon_materials"][ctype]
    cp = carbon["eps_params"]
    mp = carbon["mu_params"]
    er = complex(base["eps_real"] + cp["eps_real_slope"] * cc,
                 -(base["eps_imag"] + cp["eps_imag_slope"] * cc))
    mr = complex(base["mu_real"] + mp["mu_real_offset"] * (cc / 0.08),
                 -(base["mu_imag"] + mp["mu_imag_offset"] * (cc / 0.08)))
    dens = (1.0 - cc) * base["density"] + cc * carbon["density"]
    cost = (1.0 - cc) * 1.5 + cc * carbon["cost_proxy"]
    return er, mr, dens, cost

def main():
    task_dir = Path(__file__).resolve().parents[1]
    temp_dir = task_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    cfg = json.loads((task_dir / "references" / "problem_config.json").read_text())
    mdb = json.loads((task_dir / "references" / "material_db.json").read_text())
    freqs = np.linspace(cfg["freq_ghz_min"] * 1e9, cfg["freq_ghz_max"] * 1e9, cfg["num_freq_points"])
    w, n = cfg["weights"], cfg["normalization"]
    min_eab = cfg.get("min_eab_ghz", 0.0)

    best_score, best_sub = -1e18, None
    random.seed(42)

    for _ in range(3000):
        ctype = random.choice(cfg["valid_carbon_types"])
        cc = random.uniform(cfg["carbon_content_min"], cfg["carbon_content_max"])
        d_mm = random.uniform(cfg["d_mm_min"], cfg["d_mm_max"])

        er, mr, dens, cost = get_props(ctype, cc, mdb)
        d_m = d_mm * 1e-3
        rl = np.zeros(len(freqs))
        for i, f in enumerate(freqs):
            g = 1j * (2 * np.pi * f * d_m / C0) * np.sqrt(mr * er)
            zi = Z0 * np.sqrt(mr / er) * np.tanh(g)
            r = abs((zi - Z0) / (zi + Z0))
            rl[i] = 20 * np.log10(max(r, 1e-15))

        rl_min = float(np.min(rl))
        mask = rl <= -10; ml = cl = ei = 0
        for i, f in enumerate(mask):
            if f: cl += 1
            else: cl = 0
            if cl > ml: ml = cl; ei = i
        eab = (freqs[ei] - freqs[ei - ml + 1]) / 1e9 if ml > 0 else 0.0
        if eab < min_eab: continue

        s = (w["eab10"] * norm(eab, n["eab10_ghz"]["min"], n["eab10_ghz"]["max"])
             + w["rl_min"] * norm(abs(rl_min), n["abs_rl_min_db"]["min"], n["abs_rl_min_db"]["max"])
             - w["thickness"] * norm(d_mm, n["thickness_mm"]["min"], n["thickness_mm"]["max"])
             - w["density"] * norm(dens, n["density"]["min"], n["density"]["max"])
             - w["cost"] * norm(cost, n["cost"]["min"], n["cost"]["max"]))

        if s > best_score:
            best_score = s
            best_sub = {
                "benchmark_id": cfg["benchmark_id"],
                "carbon_type": ctype,
                "carbon_content": round(cc, 4),
                "d_mm": round(d_mm, 4),
            }

    out = temp_dir / "submission.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(best_sub, f, indent=2)
    print(f"Baseline done. Best score: {best_score:.4f}")
    print(f"Submission: {json.dumps(best_sub, indent=2)}")
    print(f"Written to {out}")

if __name__ == "__main__":
    main()
