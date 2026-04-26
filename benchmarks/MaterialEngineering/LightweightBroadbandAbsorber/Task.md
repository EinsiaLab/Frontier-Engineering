# LightweightBroadbandAbsorber — Task Specification

## 1. Background

Lightweight broadband microwave absorbers are essential in aerospace, unmanned aerial vehicles, and portable electronic systems where both electromagnetic stealth and weight reduction are critical. This benchmark is based on the CNTs@Nd₀.₁₅-BaM/PE composite system (Wang et al., *Materials* 2024, 17, 3433), where the best experimental result achieved RL_min = −58.01 dB with EAB = 4.26 GHz at 1.9 mm thickness.

The task targets the **8.2–18 GHz range** and introduces a **minimum bandwidth hard constraint** and a **heavily penalized density** to push optimizers toward lightweight solutions.

## 2. Design Variables

The optimizer controls five variables across **four material components**:

| Variable | Symbol | Unit | Range | Description |
|----------|--------|------|-------|-------------|
| Thickness | `d_mm` | mm | [1.0, 5.0] | Absorber layer thickness |
| Magnetic absorber fraction | `phi_magnetic_absorber` | — | [0, 1] | Nd₀.₁₅-BaM (density 5.1 g/cm³) |
| Conductive filler fraction | `phi_conductive_filler` | — | [0, 1] | CNTs at 8wt% (density 1.7 g/cm³) |
| Lightweight magnetic fraction | `phi_lightweight_magnetic` | — | [0, 1] | Hollow Nd-BaM (density 2.8 g/cm³) |
| Matrix fraction | `phi_matrix` | — | [0, 1] | PE matrix (density 0.95 g/cm³) |

**Constraint**: All volume fractions must sum to 1.0 (tolerance: 1e-6).

## 3. Evaluation

### 3.1 Material Property Estimation

Effective properties computed using **linear volume-fraction mixing**:

$$\varepsilon_{r,eff} = \sum_i \phi_i \cdot \varepsilon_{r,i}, \quad \mu_{r,eff} = \sum_i \phi_i \cdot \mu_{r,i}$$

> **Simplifications**: Frequency-independent constant parameters; linear mixing rule. See `material_db.json` for details. Convention: $\varepsilon_r = \varepsilon' - j\varepsilon''$ (negative imaginary part).

### 3.2 Physical Model

Standard transmission line theory with PEC backing:

$$Z_{in} = Z_0 \sqrt{\frac{\mu_r}{\varepsilon_r}} \tanh\left(j \frac{2\pi f d}{c} \sqrt{\mu_r \varepsilon_r}\right)$$

$$RL(f) = 20 \log_{10} \left| \frac{Z_{in} - Z_0}{Z_{in} + Z_0} \right|$$

### 3.3 Metrics

- **Frequency range**: 8.2–18.0 GHz (197 points)
- **$RL_{min}$**: minimum reflection loss
- **$EAB_{10}$**: maximum continuous bandwidth where $RL \leq -10\;\text{dB}$

### 3.4 Hard Constraint

**$EAB_{10} < 4.0$ GHz → infeasible** (`combined_score = 0`).

### 3.5 Scoring

All metrics min-max normalized to [0, 1]:

| Metric | Range | Unit |
|--------|-------|------|
| $EAB_{10}$ | [0, 9.8] | GHz |
| $|RL_{min}|$ | [0, 60] | dB |
| $d$ | [1.0, 5.0] | mm |
| $\rho$ | [0.9, 5.5] | g/cm³ |
| cost | [1.0, 4.0] | — |

$$\text{combined\_score} = 1.0 \cdot \hat{EAB} + 0.15 \cdot |\widehat{RL}_{min}| - 0.4 \cdot \hat{d} - 0.5 \cdot \hat{\rho} - 0.05 \cdot \widehat{cost}$$

> **Important**: Final results determined solely by `verification/evaluator.py`.

## 4. Input / Output

### 4.1 Input
- `references/material_db.json`: material database (fixed)
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
bash scripts/run_v2_unified.sh MaterialEngineering/LightweightBroadbandAbsorber \
  algorithm=openevolve \
  algorithm.iterations=0
```

## 7. References

- Wang, Y.; et al. "Preparation and microwave absorption properties of CNTs@Nd-BaM/PE composites." *Materials* 2024, 17, 3433.
