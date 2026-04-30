# MicrowaveAbsorberDesign — Task Specification

## 1. Background

Microwave absorbing materials (MAMs) are critical in electromagnetic compatibility (EMC), radar cross-section reduction, and electronic device shielding. A well-designed absorber should achieve strong absorption (low reflection loss) over a wide frequency band while remaining thin, lightweight, and cost-effective.

This benchmark targets the **X-band (8–12 GHz)**, one of the most commonly used frequency ranges for radar and satellite communication systems. The task requires optimizing a **single-layer absorber** backed by a **perfect electrical conductor (PEC)**, a standard evaluation configuration in the microwave absorption literature.

## 2. Design Variables

The optimizer controls four variables:

| Variable | Symbol | Unit | Range | Description |
|----------|--------|------|-------|-------------|
| Thickness | `d_mm` | mm | [1.0, 5.0] | Absorber layer thickness |
| Dielectric filler fraction | `phi_dielectric` | — | [0, 1] | Volume fraction of dielectric filler |
| Magnetic filler fraction | `phi_magnetic` | — | [0, 1] | Volume fraction of magnetic filler |
| Matrix fraction | `phi_matrix` | — | [0, 1] | Volume fraction of polymer matrix |

**Constraint**: `phi_dielectric + phi_magnetic + phi_matrix = 1.0` (tolerance: 1e-6).

The material properties (complex permittivity, complex permeability, density, cost proxy) for each component are fixed in `references/material_db.json`. The evaluator computes effective properties using a **linear volume-fraction mixing rule** (see Section 3.1).

## 3. Evaluation Metrics and Scoring

To ensure transparency and reproducibility, the evaluation of generated absorber designs is strictly based on standard transmission line theory.

### 3.1 Material Property Estimation

The effective electromagnetic parameters of the composite absorber are computed using a **linear volume-fraction mixing rule**:

$$\varepsilon_{r,eff} = \sum_i \phi_i \cdot \varepsilon_{r,i}, \quad \mu_{r,eff} = \sum_i \phi_i \cdot \mu_{r,i}$$

> **Note on simplifications adopted in this benchmark version:**
> - **Frequency-independent parameters**: All material properties in `material_db.json` are constant approximations. Real materials exhibit frequency-dependent dispersion (especially ferrite-type fillers in the X-band), which is not modeled in this version.
> - **Linear mixing**: The linear rule of mixtures is a first-order approximation. More accurate effective medium theories (e.g., Maxwell-Garnett, Bruggeman) account for particle shape, percolation, and interfacial effects, but are not used in this version to maintain benchmark simplicity and reproducibility.
> - **Bulk density**: The magnetic filler density (7.8 g/cm³) represents the bulk density of carbonyl iron powder. The effective composite density is computed via linear volume-fraction mixing.
>
> These simplifications are intentional for the first version of this benchmark. Future versions may introduce frequency-dependent parameters and nonlinear mixing models.

### 3.2 Physical Model: Reflection Loss Calculation

For a single-layer homogeneous microwave absorber backed by a PEC, the input impedance at the absorber surface is:

$$Z_{in} = Z_0 \sqrt{\frac{\mu_r}{\varepsilon_r}} \tanh\left(j \frac{2\pi f d}{c} \sqrt{\mu_r \varepsilon_r}\right)$$

The reflection loss (RL) is then:

$$RL(f) = 20 \log_{10} \left| \frac{Z_{in} - Z_0}{Z_{in} + Z_0} \right|$$

**Parameter definitions and conventions:**

- $Z_0 \approx 377\;\Omega$: impedance of free space.
- $\varepsilon_r = \varepsilon' - j\varepsilon''$: complex relative permittivity (**negative-imaginary-part convention**).
- $\mu_r = \mu' - j\mu''$: complex relative permeability (**negative-imaginary-part convention**).
- $f$: frequency in Hz.
- $d$: absorber thickness in **meters** (the evaluator internally converts from the submitted `d_mm`).
- $c \approx 2.998 \times 10^8\;\text{m/s}$: speed of light in vacuum.

> **Sign convention**: This benchmark strictly uses the $e^{j\omega t}$ time-harmonic convention, resulting in $\varepsilon_r = \varepsilon' - j\varepsilon''$ and $\mu_r = \mu' - j\mu''$ where $\varepsilon'' > 0$ and $\mu'' > 0$ represent losses. This convention is consistent throughout the material database, evaluator code, and this document.

### 3.3 Evaluation Metrics

The evaluator computes the RL curve on a fixed X-band frequency grid:

- **Frequency range**: 8.0 – 12.0 GHz
- **Sampling**: 161 linearly spaced frequency points

From the computed RL curve, two primary metrics are extracted:

- **Minimum Reflection Loss** ($RL_{min}$): the minimum RL value within the evaluation band. More negative values indicate better peak absorption.
- **Effective Absorption Bandwidth** ($EAB_{10}$): the **maximum continuous** bandwidth span (in GHz) over which $RL \leq -10\;\text{dB}$, a commonly used criterion for effective microwave absorption.

In addition, the evaluator computes two auxiliary engineering proxies from the predefined material database and mixture rules:

- $\rho$: effective density (g/cm³), computed via linear mixing of component densities.
- $\text{cost}$: dimensionless manufacturing cost proxy, computed via linear mixing of component cost proxies.

### 3.4 Final Scoring

The final benchmark objective is a single scalar `combined_score` (higher is better).

**All metrics are first normalized to [0, 1] using min-max scaling** with predefined physically reasonable ranges (specified in `references/problem_config.json`):

| Metric | Range | Unit |
|--------|-------|------|
| $EAB_{10}$ | [0, 4.0] | GHz |
| $|RL_{min}|$ | [0, 30.0] | dB |
| $d$ | [1.0, 5.0] | mm |
| $\rho$ | [1.0, 8.0] | g/cm³ |
| cost | [1.0, 3.0] | — |

The normalized scoring formula is:

$$\text{combined\_score} = w_1 \cdot \hat{EAB}_{10} + w_2 \cdot |\widehat{RL}_{min}| - w_3 \cdot \hat{d} - w_4 \cdot \hat{\rho} - w_5 \cdot \widehat{cost}$$

where $\hat{x}$ denotes the min-max normalized value of $x$, and the weights are:

| Weight | Value | Description |
|--------|-------|-------------|
| $w_1$ (eab10) | 1.0 | Bandwidth reward (dominant) |
| $w_2$ (rl_min) | 0.2 | Absorption depth reward |
| $w_3$ (thickness) | 0.5 | Thickness penalty (elevated for lightweight applications) |
| $w_4$ (density) | 0.1 | Density penalty |
| $w_5$ (cost) | 0.05 | Cost penalty |

> **Important**: The equations above describe the intended physical model and scoring principles. However, the final benchmark result is determined solely by the official implementation in `verification/evaluator.py`. In case of any discrepancy caused by numerical precision, discretization, boundary handling, or unit conversion, the evaluator output should be treated as the ground truth.

## 4. Input / Output Format

### 4.1 Input

The candidate program has access to:

- `references/material_db.json`: material property database (fixed, read-only)
- `references/problem_config.json`: benchmark configuration (fixed, read-only)

### 4.2 Output

The candidate program must write a JSON file to `temp/submission.json` with the following schema:

```json
{
  "benchmark_id": "microwave_absorber_single_layer_xband",
  "d_mm": 2.5,
  "phi_dielectric": 0.20,
  "phi_magnetic": 0.35,
  "phi_matrix": 0.45
}
```

All values must be finite numbers. Volume fractions must be non-negative and sum to 1.0.

## 5. Feasibility Rules

A submission is marked as **infeasible** (`valid=0`, `combined_score=0`) if any of the following conditions is met:

1. `submission.json` is missing or cannot be parsed as valid JSON.
2. Any required key is absent.
3. `benchmark_id` does not match the expected value.
4. `d_mm` is not a finite number or falls outside [1.0, 5.0].
5. Any volume fraction is not a finite number or falls outside [0, 1].
6. Volume fractions do not sum to 1.0 within the specified tolerance (1e-6).
7. The candidate program times out (120-second limit) or exits with a non-zero return code.

## 6. How to Run

```bash
# From the MicrowaveAbsorberDesign/ directory:

# Test the minimal initialization
python verification/evaluator.py scripts/init.py

# Test the baseline
python verification/evaluator.py baseline/solution.py

# Mainline unified compatibility check
bash scripts/run_v2_unified.sh MaterialEngineering/MicrowaveAbsorberDesign \
  algorithm=openevolve \
  algorithm.iterations=0
```
