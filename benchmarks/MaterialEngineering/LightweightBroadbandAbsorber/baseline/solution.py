"""
<<<<<<< Updated upstream
Baseline for LightweightBroadbandAbsorber. Random search, 3000 samples.
=======
Baseline for LightweightBroadbandAbsorber. Random search, 2000 samples.
>>>>>>> Stashed changes
"""
import json, random
from pathlib import Path
import numpy as np

Z0, C0 = 377.0, 2.998e8

def norm(v, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

def main():
    task_dir = Path(__file__).resolve().parents[1]
    temp_dir = task_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    cfg = json.loads((task_dir / "references" / "problem_config.json").read_text())
    mdb = json.loads((task_dir / "references" / "material_db.json").read_text())
<<<<<<< Updated upstream
    freqs = np.linspace(cfg["freq_ghz_min"] * 1e9, cfg["freq_ghz_max"] * 1e9, cfg["num_freq_points"])
    w, n = cfg["weights"], cfg["normalization"]
    mat = mdb["matrix"]; mag = mdb["magnetic_absorber"]
    cnt = mdb["conductive_filler"]; lm = mdb["lightweight_magnetic"]
=======
    freqs = np.linspace(cfg["freq_ghz_min"]*1e9, cfg["freq_ghz_max"]*1e9, cfg["num_freq_points"])
    w, n = cfg["weights"], cfg["normalization"]
    mat, die, mag, lmg = mdb["matrix"], mdb["dielectric_filler"], mdb["magnetic_filler"], mdb["lightweight_magnetic_filler"]
>>>>>>> Stashed changes
    min_eab = cfg.get("min_eab_ghz", 0.0)

    best_score, best_sub = -1e18, None
    random.seed(42)

<<<<<<< Updated upstream
    for _ in range(3000):
        pm = random.uniform(0.05, 0.50)
        pc = random.uniform(0.02, 0.30)
        pl = random.uniform(0.0, 0.30)
        if pm + pc + pl > 0.92: continue
        px = 1.0 - pm - pc - pl
        d_mm = random.uniform(cfg["d_mm_min"], cfg["d_mm_max"])

        comps = [(px, mat), (pm, mag), (pc, cnt), (pl, lm)]
        er = complex(sum(p * c["eps_real"] for p, c in comps),
                     -sum(p * c["eps_imag"] for p, c in comps))
        mr = complex(sum(p * c["mu_real"] for p, c in comps),
                     -sum(p * c["mu_imag"] for p, c in comps))
        dens = sum(p * c["density"] for p, c in comps)
        cost = sum(p * c["cost_proxy"] for p, c in comps)
=======
    for _ in range(2000):
        pd = random.uniform(0.1, 0.6)
        pm = random.uniform(0.0, 0.4)
        plm = random.uniform(0.0, 0.5)
        if pd + pm + plm > 0.95: continue
        px = 1.0 - pd - pm - plm
        d_mm = random.uniform(cfg["d_mm_min"], cfg["d_mm_max"])

        comps = [(px, mat), (pd, die), (pm, mag), (plm, lmg)]
        er = complex(sum(p*c["eps_real"] for p,c in comps), -sum(p*c["eps_imag"] for p,c in comps))
        mr = complex(sum(p*c["mu_real"] for p,c in comps), -sum(p*c["mu_imag"] for p,c in comps))
        dens = sum(p*c["density"] for p,c in comps)
        cost = sum(p*c["cost_proxy"] for p,c in comps)
>>>>>>> Stashed changes

        d_m = d_mm * 1e-3
        rl = np.zeros(len(freqs))
        for i, f in enumerate(freqs):
<<<<<<< Updated upstream
            g = 1j * (2 * np.pi * f * d_m / C0) * np.sqrt(mr * er)
            zi = Z0 * np.sqrt(mr / er) * np.tanh(g)
            r = abs((zi - Z0) / (zi + Z0))
            rl[i] = 20 * np.log10(max(r, 1e-15))
=======
            g = 1j*(2*np.pi*f*d_m/C0)*np.sqrt(mr*er)
            zi = Z0*np.sqrt(mr/er)*np.tanh(g)
            r = abs((zi-Z0)/(zi+Z0))
            rl[i] = 20*np.log10(max(r, 1e-15))
>>>>>>> Stashed changes

        rl_min = float(np.min(rl))
        mask = rl <= -10; ml = cl = ei = 0
        for i, f in enumerate(mask):
            if f: cl += 1
            else: cl = 0
            if cl > ml: ml = cl; ei = i
<<<<<<< Updated upstream
        eab = (freqs[ei] - freqs[ei - ml + 1]) / 1e9 if ml > 0 else 0.0
        if eab < min_eab: continue

        s = (w["eab10"] * norm(eab, n["eab10_ghz"]["min"], n["eab10_ghz"]["max"])
             + w["rl_min"] * norm(abs(rl_min), n["abs_rl_min_db"]["min"], n["abs_rl_min_db"]["max"])
             - w["thickness"] * norm(d_mm, n["thickness_mm"]["min"], n["thickness_mm"]["max"])
             - w["density"] * norm(dens, n["density"]["min"], n["density"]["max"])
             - w["cost"] * norm(cost, n["cost"]["min"], n["cost"]["max"]))
=======
        eab = (freqs[ei] - freqs[ei-ml+1]) / 1e9 if ml > 0 else 0.0

        if eab < min_eab: continue

        s = (w["eab10"]*norm(eab, n["eab10_ghz"]["min"], n["eab10_ghz"]["max"])
             + w["rl_min"]*norm(abs(rl_min), n["abs_rl_min_db"]["min"], n["abs_rl_min_db"]["max"])
             - w["thickness"]*norm(d_mm, n["thickness_mm"]["min"], n["thickness_mm"]["max"])
             - w["density"]*norm(dens, n["density"]["min"], n["density"]["max"])
             - w["cost"]*norm(cost, n["cost"]["min"], n["cost"]["max"]))
>>>>>>> Stashed changes

        if s > best_score:
            best_score = s
            best_sub = {
                "benchmark_id": cfg["benchmark_id"],
                "d_mm": round(d_mm, 4),
<<<<<<< Updated upstream
                "phi_magnetic_absorber": round(pm, 4),
                "phi_conductive_filler": round(pc, 4),
                "phi_lightweight_magnetic": round(pl, 4),
=======
                "phi_dielectric": round(pd, 4),
                "phi_magnetic": round(pm, 4),
                "phi_lightweight_magnetic": round(plm, 4),
>>>>>>> Stashed changes
                "phi_matrix": round(px, 4),
            }

    if best_sub:
<<<<<<< Updated upstream
        best_sub["phi_matrix"] = round(1.0 - best_sub["phi_magnetic_absorber"]
                                       - best_sub["phi_conductive_filler"]
                                       - best_sub["phi_lightweight_magnetic"], 6)
=======
        best_sub["phi_matrix"] = round(1.0 - best_sub["phi_dielectric"] - best_sub["phi_magnetic"] - best_sub["phi_lightweight_magnetic"], 6)
>>>>>>> Stashed changes

    out = temp_dir / "submission.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(best_sub, f, indent=2)
    print(f"Baseline done. Best score: {best_score:.4f}")
    print(f"Submission: {json.dumps(best_sub, indent=2)}")
    print(f"Written to {out}")

if __name__ == "__main__":
    main()
