# LightweightBroadbandAbsorber — Task Specification

## 1. Background

Lightweight broadband microwave absorbers are essential in aerospace stealth, EMC shielding, and portable electronics where both electromagnetic attenuation and weight reduction are critical. The CNTs@Nd-doped BaM/PE composite system, combining the magnetoelectric synergy of rare-earth-doped barium ferrite with the conductive network of carbon nanotubes in a polyethylene matrix, represents a promising approach to achieving broadband absorption with reduced density.

This benchmark is inspired by experimental work on CNTs@Nd₀.₁₅-BaM/PE composites (Wang et al., *Materials* 2024, 17, 3433), where the 2:1 PE-to-absorber ratio achieved RL_min = -58.01 dB and EAB = 4.26 GHz at only 1.9 mm thickness. The task targets the **8.2–18 GHz band** (matching standard waveguide VNA measurements) and introduces a **minimum bandwidth hard constraint** with a **heavily penalized density** to push optimizers toward lightweight solutions.

## 2. Design Variables

The optimizer controls five variables across **four material components**:

| Variable | Symbol | Unit | Range | Description |
|----------|--------|------|-------|-------------|
| Thickness | `d_mm` | mm | [1.0, 5.0] | Absorber layer thickness |
| Magnetic absorber fraction | `phi_magnetic_absorber` | — | [0, 1] | Nd₀.₁₅-BaM (rare-earth doped barium ferrite, 5.1 g/cm³) |
| Conductive filler fraction | `phi_conductive_filler` | — | [0, 1] | CNTs (carbon nanotubes, 1.7 g/cm³) |
| Lightweight magnetic fraction | `phi_lightweight_magnetic` | — | [0, 1] | Hollow Nd-BaM microspheres (2.8 g/cm³) |
| Matrix fraction | `phi_matrix` | — | [0, 1] | PE polyethylene matrix (0.95 g/cm³) |

**Constraint**: all volume fractions sum to 1.0 (tolerance: 1e-6).

### Key Design Trade-offs

- **Nd-BaM** provides magnetic loss (natural ferromagnetic resonance, eddy current loss) and enhances magneto-crystalline anisotropy via Nd³⁺ doping, but has high density (5.1 g/cm³).
- **CNTs** form conductive networks on BaM surfaces, providing dielectric loss via electronic polarization and resistive heating. They are lightweight (1.7 g/cm³) but do not contribute magnetic loss.
- **Hollow Nd-BaM microspheres** offer a lightweight magnetic alternative (2.8 g/cm³) with reduced magnetic response.
- **PE matrix** is the lightest component (0.95 g/cm³) but provides no electromagnetic absorption. Excess PE leads to impedance mismatch and reduced absorption.
- The optimizer must balance magnetoelectric synergy (CNTs + Nd-BaM) against weight (more PE = lighter but weaker absorption).

## 3. Evaluation Metrics and Scoring

### 3.1 Material Property Estimation

Effective properties are computed using **linear volume-fraction mixing**:

$$\varepsilon_{r,eff} = \sum_i \phi_i \cdot \varepsilon_{r,i}, \quad \mu_{r,eff} = \sum_i \phi_i \cdot \mu_{r,i}$$

> **Simplifications**: Frequency-independent constant parameters approximating VNA-measured averages across 8.2-18 GHz. Linear mixing rule. Convention: $\varepsilon_r = \varepsilon' - j\varepsilon''$ (negative imaginary part). See `material_db.json` for parameter values and data sources.

### 3.2 Physical Model: Reflection Loss

Single-layer transmission line theory with PEC backing:

$$Z_{in} = Z_0 \sqrt{\frac{\mu_r}{\varepsilon_r}} \tanh\left(j \frac{2\pi f d}{c} \sqrt{\mu_r \varepsilon_r}\right)$$

$$RL(f) = 20 \log_{10} \left| \frac{Z_{in} - Z_0}{Z_{in} + Z_0} \right|$$

### 3.3 Evaluation Metrics

- **Frequency range**: 8.2 – 18.0 GHz (197 linearly spaced points)
- **$RL_{min}$**: minimum RL in the evaluation band
- **$EAB_{10}$**: maximum continuous bandwidth where $RL \leq -10\;\text{dB}$

### 3.4 Hard Constraint

**Designs with $EAB_{10} < 4.0\;\text{GHz}$ are infeasible** (`feasible=0`, `combined_score=0`).

### 3.5 Final Scoring

All metrics **min-max normalized to [0, 1]**:

| Metric | Range | Unit |
|--------|-------|------|
| $EAB_{10}$ | [0, 9.8] | GHz |
| $|RL_{min}|$ | [0, 60.0] | dB |
| $d$ | [1.0, 5.0] | mm |
| $\rho$ | [0.9, 5.5] | g/cm³ |
| cost | [1.0, 4.0] | — |

$$\text{combined\_score} = w_1 \cdot \hat{EAB}_{10} + w_2 \cdot |\widehat{RL}_{min}| - w_3 \cdot \hat{d} - w_4 \cdot \hat{\rho} - w_5 \cdot \widehat{cost}$$

| Weight | Value | Description |
|--------|-------|-------------|
| $w_1$ (eab10) | 1.0 | Bandwidth reward (dominant) |
| $w_2$ (rl_min) | 0.15 | Absorption depth reward |
| $w_3$ (thickness) | 0.4 | Thickness penalty |
| $w_4$ (density) | **0.5** | **Density penalty (dominant — lightweight focus)** |
| $w_5$ (cost) | 0.05 | Cost penalty |

> **Important**: Final results determined solely by `verification/evaluator.py`.

## 4. Input / Output Format

### 4.1 Input
- `references/material_db.json`: 4-component material database (fixed)
- `references/problem_config.json`: configuration (fixed)

### 4.2 Output
`temp/submission.json`:
```json
{
  "benchmark_id": "lightweight_broadband_absorber_8_18ghz",
  "d_mm": 1.9,
  "phi_magnetic_absorber": 0.25,
  "phi_conductive_filler": 0.10,
  "phi_lightweight_magnetic": 0.05,
  "phi_matrix": 0.60
}
```

## 5. Feasibility Rules

Infeasible if:
1. `submission.json` missing or unparseable.
2. Any required key absent.
3. `benchmark_id` mismatch.
4. `d_mm` outside [1.0, 5.0] or non-finite.
5. Any volume fraction outside [0, 1] or non-finite.
6. Volume fractions do not sum to 1.0 (tolerance: 1e-6).
7. **$EAB_{10} < 4.0\;\text{GHz}$**.
8. Timeout (120s) or non-zero exit code.

## 6. How to Run

```bash
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
python -m frontier_eval task=LightweightBroadbandAbsorber algorithm.iterations=0
```

## 7. References

- Wang, C.; Feng, X.; Yu, C.; et al. "Investigating Enhanced Microwave Absorption of CNTs@Nd0.15-BaM/PE Plate via Low-Temperature Sintering and High-Energy Ball Milling." *Materials* 2024, 17, 3433.
