# Particle Physics Engineering: Non-uniform PET Geometry and Spatial Resource Pareto Optimization

## 1. Task Background

Positron Emission Tomography (PET) is a crucial molecular imaging modality. Its core physical mechanism involves detecting positrons emitted by radioactive tracers during $\beta^+$ decay. These positrons immediately undergo **antimatter annihilation** with surrounding electrons, releasing two **511 keV** gamma-ray photons traveling in almost exactly opposite directions (180°).

To capture these high-energy photons, a PET scanner is constructed with an array of extremely expensive LYSO (Lutetium-yttrium oxyorthosilicate) scintillation crystal rings.

## 2. The Engineering "Reality Gap" & Non-uniform Resource Allocation

As the Chief Hardware Architect for a next-generation PET system, your task is a classic **algorithmic spatial resource allocation problem**.

To maximize the use of a strictly limited LYSO crystal budget (e.g., $15,000,000 \text{ mm}^3$), you can no longer design a scanner with completely uniform crystal thickness. **Rings closer to the center of the Field of View (FOV) have a significantly larger solid angle coverage (capturing more photons), while the capture efficiency drops exponentially for edge rings.**

You must write a heuristic algorithm to dynamically allocate geometric parameters for **20 independent detector rings** aligned along the axial z-axis, finding the Pareto optimal solution among:

1. **System Sensitivity**: Allocate thicker crystals to central rings to stop highly penetrating 511 keV gamma rays.
2. **Parallax Error (DOI Effect)**: Thicker crystals (large $H$) combined with a smaller detector ring (small $R$) significantly increase the depth-of-interaction (DOI) uncertainty, causing spatial resolution degradation.
3. **Strict Economic Budget**: LYSO crystal volume increases geometrically. You must write logic to make trade-offs, thinning out edge rings to preserve the budget for central ones.

## 3. Decision Variables

The AI Agent must output a file named `solution.json`. This file must be a **JSON Array containing 20 objects**.
Each object represents a single detector ring (with a fixed axial width of 10 mm) and must contain the following continuous geometric variables (all units in mm):

* `ring_id`: The index of the ring (from 0 to 19).
* `R` (Inner radius of the ring). Search space: $300.0$ to $500.0$
* `H` (Radial thickness of the LYSO crystals). Search space: $10.0$ to $30.0$
* `W` (Cross-sectional width of a single crystal). Search space: $2.0$ to $6.0$

## 4. Physics Constraints & Scoring Logic

The evaluator script (`evaluator.py`) iterates over your 20 rings and accumulates scores based on the following physical models:

### A. Crystal Volume & Cost Penalty
The volume of each ring is $V_i = \pi \cdot ((R_i+H_i)^2 - R_i^2) \cdot 10.0$.
If the total crystal volume $\sum V_i$ exceeds the budget constraint ($15,000,000 \text{ mm}^3$), a massive negative penalty is applied.

### B. Absolute Sensitivity Gain
Let the axial distance of the $i$-th ring from the scanner's center be $z_i = (i - 9.5) \cdot 10.0$.
The solid angle factor is inversely proportional to its straight-line distance to the center, $d_i = \sqrt{R_i^2 + z_i^2}$. Combined with the exponential attenuation of 511 keV gamma rays in LYSO, the total sensitivity score is proportional to:
$$S_{total} = \sum_{i=0}^{19} \left( \frac{10.0}{\sqrt{R_i^2 + z_i^2}} \cdot (1 - e^{-0.087 H_i})^2 \right)$$

### C. Spatial Resolution Penalty
The overall spatial resolution penalty is the average Parallax Error ($\Gamma_{avg}$) across all 20 rings:
$$\Gamma_i = \sqrt{W_i^2 + \left(200.0 \cdot \frac{H_i}{R_i}\right)^2}$$
A larger average $\Gamma$ results in a heavier score deduction.

## 5. Output Format Requirement

The Agent must generate a valid `solution.json` in the current working directory, formatted as an array of 20 ring objects:

```json
[
  {
    "ring_id": 0,
    "R": 400.0,
    "H": 10.0,
    "W": 4.0
  },
  ... (total 20 objects) ...
  {
    "ring_id": 19,
    "R": 400.0,
    "H": 10.0,
    "W": 4.0
  }
]
```
**Goal**: Write an intelligent allocation heuristic to maximize the comprehensive score (Score = Sensitivity Gain - Resolution Penalty - Cost Penalty) without violating the total volume budget.