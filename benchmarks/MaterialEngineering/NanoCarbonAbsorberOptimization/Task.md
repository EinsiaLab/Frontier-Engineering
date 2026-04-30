# NanoCarbonAbsorberOptimization — Task Specification

## 1. Background

The type and content of nano-carbon materials critically influence the microwave absorption performance of ferrite-based composites. Carbon nanotubes (CNTs), graphene oxide (GO), and onion-like carbon (OLC) provide fundamentally different dielectric-loss mechanisms.

This benchmark is based on the Nd₀.₁₅-BaM/NC composite system (Feng et al., *J Mater Sci: Mater Eng* 2024, 19:49) and targets the **2–18 GHz** band. The task is a **mixed-variable optimization** problem: select the best carbon material type (discrete) and jointly optimize carbon content and absorber thickness (continuous).

## 2. Design Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `carbon_type` | Discrete | `"CNTs"`, `"GO"`, `"OLC"` | Nano-carbon material selection |
| `carbon_content` | Continuous | `[0.01, 0.10]` | Mass fraction of nano-carbon in the composite |
| `d_mm` | Continuous | `[1.5, 5.0]` mm | Absorber thickness |

## 3. Evaluation

### 3.1 Effective Property Model

The composite's effective parameters depend on the selected carbon type and content:

`eps_eff = eps_base + slope * carbon_content`

with carbon-type-specific parameters in `references/material_db.json`.

### 3.2 Metrics

- Frequency range: 2.0–18.0 GHz (321 points)
- `RL_min`: minimum reflection loss
- `EAB_10`: maximum continuous bandwidth where `RL <= -10 dB`

### 3.3 Hard Constraint

`EAB_10 < 3.0 GHz` is infeasible and yields `combined_score = 0`.

### 3.4 Final Score

All metrics are min-max normalized to `[0, 1]` and combined as:

`combined_score = reward(EAB_10, |RL_min|) - penalty(thickness, density, cost)`

The evaluator implementation in `verification/evaluator.py` is the ground truth.

## 4. Output Contract

The candidate must write `temp/submission.json`:

```json
{
  "benchmark_id": "nanocarbon_absorber_optimization_2_18ghz",
  "carbon_type": "CNTs",
  "carbon_content": 0.04,
  "d_mm": 1.5
}
```

## 5. Validity Rules

A submission is invalid if:

- output is missing or malformed
- required keys are absent
- `benchmark_id` mismatches
- `carbon_type` is not one of `"CNTs"`, `"GO"`, `"OLC"`
- `carbon_content` is non-finite or outside `[0.01, 0.10]`
- `d_mm` is non-finite or outside `[1.5, 5.0]`
- `EAB_10 < 3.0 GHz`
- the candidate times out or exits non-zero
