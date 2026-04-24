"""
Baseline solution for MicrowaveAbsorberDesign benchmark.
Uses random search over 500 samples to find a reasonable design.
"""
import json
import random
from pathlib import Path

import numpy as np

Z0 = 377.0
C0 = 2.998e8


def normalize(value, vmin, vmax):
    if vmax <= vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def compute_rl_and_eab(eps_r, mu_r, d_mm, freqs_hz, threshold_db=-10.0):
    d_m = d_mm * 1e-3
    rl_db = np.zeros(len(freqs_hz))
    for i, freq_hz in enumerate(freqs_hz):
        gamma = 1j * (2.0 * np.pi * freq_hz * d_m / C0) * np.sqrt(mu_r * eps_r)
        z_in = Z0 * np.sqrt(mu_r / eps_r) * np.tanh(gamma)
        refl = abs((z_in - Z0) / (z_in + Z0))
        rl_db[i] = 20.0 * np.log10(max(refl, 1e-15))

    rl_min = float(np.min(rl_db))
    mask = rl_db <= threshold_db
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
        eab10 = 0.0
    else:
        start_idx = end_idx - max_len + 1
        eab10 = (freqs_hz[end_idx] - freqs_hz[start_idx]) / 1e9
    return rl_min, eab10


def main():
    task_dir = Path(__file__).resolve().parents[1]
    temp_dir = task_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    config = json.loads((task_dir / "references" / "problem_config.json").read_text())
    matdb = json.loads((task_dir / "references" / "material_db.json").read_text())

    freqs_hz = np.linspace(
        config["freq_ghz_min"] * 1e9,
        config["freq_ghz_max"] * 1e9,
        config["num_freq_points"],
    )
    weights = config["weights"]
    norm = config["normalization"]
    mat = matdb["matrix"]
    die = matdb["dielectric_filler"]
    mag = matdb["magnetic_filler"]

    best_score = -1e18
    best_sub = None
    random.seed(42)

    for _ in range(500):
        phi_d = random.uniform(0.05, 0.50)
        phi_m = random.uniform(0.05, 0.50)
        phi_x = 1.0 - phi_d - phi_m
        if phi_x < 0.05:
            continue
        d_mm = random.uniform(config["d_mm_min"], config["d_mm_max"])

        eps_real = phi_x * mat["eps_real"] + phi_d * die["eps_real"] + phi_m * mag["eps_real"]
        eps_imag = phi_x * mat["eps_imag"] + phi_d * die["eps_imag"] + phi_m * mag["eps_imag"]
        mu_real = phi_x * mat["mu_real"] + phi_d * die["mu_real"] + phi_m * mag["mu_real"]
        mu_imag = phi_x * mat["mu_imag"] + phi_d * die["mu_imag"] + phi_m * mag["mu_imag"]
        density = phi_x * mat["density"] + phi_d * die["density"] + phi_m * mag["density"]
        cost = phi_x * mat["cost_proxy"] + phi_d * die["cost_proxy"] + phi_m * mag["cost_proxy"]

        rl_min, eab10 = compute_rl_and_eab(
            complex(eps_real, -eps_imag),
            complex(mu_real, -mu_imag),
            d_mm,
            freqs_hz,
        )
        score = (
            weights["eab10"] * normalize(eab10, norm["eab10_ghz"]["min"], norm["eab10_ghz"]["max"])
            + weights["rl_min"]
            * normalize(abs(rl_min), norm["abs_rl_min_db"]["min"], norm["abs_rl_min_db"]["max"])
            - weights["thickness"]
            * normalize(d_mm, norm["thickness_mm"]["min"], norm["thickness_mm"]["max"])
            - weights["density"] * normalize(density, norm["density"]["min"], norm["density"]["max"])
            - weights["cost"] * normalize(cost, norm["cost"]["min"], norm["cost"]["max"])
        )
        if score > best_score:
            best_score = score
            best_sub = {
                "benchmark_id": config["benchmark_id"],
                "d_mm": round(d_mm, 4),
                "phi_dielectric": round(phi_d, 4),
                "phi_magnetic": round(phi_m, 4),
                "phi_matrix": round(phi_x, 4),
            }

    best_sub["phi_matrix"] = round(
        1.0 - best_sub["phi_dielectric"] - best_sub["phi_magnetic"], 6
    )
    output_path = temp_dir / "submission.json"
    output_path.write_text(json.dumps(best_sub, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline search completed. Best score proxy: {best_score:.4f}")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
