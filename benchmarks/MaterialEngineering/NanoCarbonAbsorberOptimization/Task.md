# NanoCarbonAbsorberOptimization — Task Specification

## 1. Background

The type and content of nano-carbon (NC) materials critically influence the microwave absorption performance of ferrite-based composites. Different NC materials — carbon nanotubes (CNTs), graphene oxide (GO), and onion-like carbon (OLC) — provide fundamentally different loss mechanisms: CNTs form conductive networks for strong resistive loss, GO offers polarization loss via oxygen-containing functional groups, and OLC provides moderate conductivity through its partially graphitized shell.

This benchmark is based on the Nd₀.₁₅-BaM/NC composite system (Feng et al., *J Mater Sci: Mater Eng* 2024, 19:49), where Nd₀.₁₅-BaM/8%CNTs achieved RL_min = −123.12 dB at 7.97 GHz with EAB = 5.46 GHz at 2.5 mm thickness. The task targets the **2–18 GHz band** and introduces a **mixed-variable optimization** problem: the optimizer must jointly select the best carbon material type (discrete) and optimize its content and absorber thickness (continuous).

## 2. Design Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `carbon_type` | **Discrete** | "CNTs", "GO", "OLC" | Nano-carbon material selection |
| `carbon_content` | Continuous | [0.01, 0.10] | Mass fraction of NC in composite |
| `d_mm` | Continuous | [1.5, 5.0] mm | Absorber layer thickness |

### Key Design Trade-offs

- **CNTs** (sheet resistivity ~1.32 Ω/sq): Highest conductivity, strongest dielectric loss, forms conductive network on Nd-BaM particles. But excessive content causes impedance mismatch.
- **GO** (sheet resistivity ~1.55×10⁹ Ω/sq): Rich functional groups provide polarization loss. Low-frequency EAB complements BaM's high-frequency absorption. But weak overall dielectric loss.
- **OLC** (sheet resistivity ~9.67 Ω/sq): Spherical particles with moderate conductivity. Cannot form complete conductive network, limiting performance. But offers a balance between CNTs and GO.
- **Content**: Too low = weak electromagnetic coupling. Too high = impedance mismatch and agglomeration.
- **Thickness**: Governs quarter-wavelength matching conditions and absorption peak frequency.

## 3. Evaluation

### 3.1 Effective Property Model

The composite's effective electromagnetic parameters depend on the selected carbon type and content:

$$\varepsilon'_{eff} = \varepsilon'_{base} + k_{\varepsilon'} \cdot \phi_c, \quad \varepsilon''_{eff} = \varepsilon''_{base} + k_{\varepsilon''} \cdot \phi_c$$

where $\varepsilon'_{base}$, $\varepsilon''_{base}$ are the Nd-BaM base permittivity, $k$ values are carbon-type-specific slopes fitted from VNA data, and $\phi_c$ is carbon content.

> **Simplifications**: Frequency-independent parameters, linear content dependence. See `material_db.json` for per-carbon-type parameters. Convention: $\varepsilon_r = \varepsilon' - j\varepsilon''$.

### 3.2 Physical Model

Standard transmission line theory with PEC backing (same as other MaterialEngineering tasks).

### 3.3 Metrics

- **Frequency**: 2.0–18.0 GHz (321 points)
- **$RL_{min}$**: minimum reflection loss
- **$EAB_{10}$**: maximum continuous bandwidth where $RL \leq -10$ dB

### 3.4 Hard Constraint

**$EAB_{10} < 3.0$ GHz → infeasible** (`combined_score = 0`).

### 3.5 Scoring

All metrics min-max normalized to [0, 1]:

| Metric | Range | Unit |
|--------|-------|------|
| $EAB_{10}$ | [0, 16] | GHz |
| $|RL_{min}|$ | [0, 130] | dB |
| $d$ | [1.5, 5.0] | mm |
| $\rho$ | [3.0, 5.5] | g/cm³ |
| cost | [1.0, 4.0] | — |

$$\text{combined\_score} = 1.0 \cdot \hat{EAB} + 0.2 \cdot |\widehat{RL}_{min}| - 0.3 \cdot \hat{d} - 0.15 \cdot \hat{\rho} - 0.05 \cdot \widehat{cost}$$

> **Important**: Final results determined solely by `verification/evaluator.py`.

## 4. Input / Output

### 4.1 Input
- `references/material_db.json`: NC parameter models (fixed)
- `references/problem_config.json`: configuration (fixed)

### 4.2 Output
`temp/submission.json`:
```json
{
  "benchmark_id": "nanocarbon_absorber_optimization_2_18ghz",
  "carbon_type": "CNTs",
  "carbon_content": 0.04,
  "d_mm": 1.5
}
```

## 5. Feasibility Rules

Infeasible if:
1. `submission.json` missing or unparseable.
2. Any required key absent.
3. `benchmark_id` mismatch.
4. `carbon_type` not in ["CNTs", "GO", "OLC"].
5. `carbon_content` outside [0.01, 0.10] or non-finite.
6. `d_mm` outside [1.5, 5.0] or non-finite.
7. **$EAB_{10} < 3.0$ GHz**.
8. Timeout (120s) or non-zero exit code.

## 6. How to Run

```bash
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
python -m frontier_eval task=NanoCarbonAbsorberOptimization algorithm.iterations=0
```

## 7. References

- Feng, X.; Li, P.; Yu, H.; et al. "Incorporation of nanocarbon materials of various dimensions enhances the microwave absorption properties of Nd-doped barium ferrite." *J Mater Sci: Mater Eng* 2024, 19:49.
