"""
Baseline for NanoCarbonAbsorberOptimization.
Searches across all three carbon types with random content and thickness.
"""
import json
import random
from pathlib import Path

import numpy as np

Z0, C0 = 377.0, 2.998e8


def norm(v, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def get_props(ctype, cc, mdb):
    base = mdb["base_absorber"]
    carbon = mdb["carbon_materials"][ctype]
    cp = carbon["eps_params"]
    mp = carbon["mu_params"]
    er = complex(
        base["eps_real"] + cp["eps_real_slope"] * cc,
        -(base["eps_imag"] + cp["eps_imag_slope"] * cc),
    )
    mr = complex(
        base["mu_real"] + mp["mu_real_offset"] * (cc / 0.08),
        -(base["mu_imag"] + mp["mu_imag_offset"] * (cc / 0.08)),
    )
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
    weights, norm_cfg = cfg["weights"], cfg["normalization"]
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
        for i, freq_hz in enumerate(freqs):
            gamma = 1j * (2 * np.pi * freq_hz * d_m / C0) * np.sqrt(mr * er)
            z_in = Z0 * np.sqrt(mr / er) * np.tanh(gamma)
            refl = abs((z_in - Z0) / (z_in + Z0))
            rl[i] = 20 * np.log10(max(refl, 1e-15))

        rl_min = float(np.min(rl))
        mask = rl <= -10
        max_len = cur_len = end_idx = 0
        for i, flag in enumerate(mask):
            if flag:
                cur_len += 1
            else:
                cur_len = 0
            if cur_len > max_len:
                max_len = cur_len
                end_idx = i
        eab = (freqs[end_idx] - freqs[end_idx - max_len + 1]) / 1e9 if max_len > 0 else 0.0
        if eab < min_eab:
            continue

        score = (
            weights["eab10"] * norm(eab, norm_cfg["eab10_ghz"]["min"], norm_cfg["eab10_ghz"]["max"])
            + weights["rl_min"]
            * norm(abs(rl_min), norm_cfg["abs_rl_min_db"]["min"], norm_cfg["abs_rl_min_db"]["max"])
            - weights["thickness"]
            * norm(d_mm, norm_cfg["thickness_mm"]["min"], norm_cfg["thickness_mm"]["max"])
            - weights["density"] * norm(dens, norm_cfg["density"]["min"], norm_cfg["density"]["max"])
            - weights["cost"] * norm(cost, norm_cfg["cost"]["min"], norm_cfg["cost"]["max"])
        )

        if score > best_score:
            best_score = score
            best_sub = {
                "benchmark_id": cfg["benchmark_id"],
                "carbon_type": ctype,
                "carbon_content": round(cc, 4),
                "d_mm": round(d_mm, 4),
            }

    out = temp_dir / "submission.json"
    out.write_text(json.dumps(best_sub, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline done. Best score: {best_score:.4f}")
    print(f"Submission: {json.dumps(best_sub, indent=2)}")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
