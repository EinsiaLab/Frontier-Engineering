# MicrowaveAbsorberDesign — Task Specification

## 1. Background

Microwave absorbing materials are critical for electromagnetic compatibility, radar cross-section reduction, and shielding. This benchmark targets a **single-layer X-band (8-12 GHz)** absorber backed by a perfect electrical conductor.

## 2. Design Variables

The optimizer controls:

- `d_mm`: absorber thickness in mm, range `[1.0, 5.0]`
- `phi_dielectric`: dielectric filler fraction, range `[0, 1]`
- `phi_magnetic`: magnetic filler fraction, range `[0, 1]`
- `phi_matrix`: matrix fraction, range `[0, 1]`

Constraint:

- `phi_dielectric + phi_magnetic + phi_matrix = 1.0` within tolerance `1e-6`

## 3. Scoring

The evaluator computes effective electromagnetic properties by linear volume-fraction mixing and then evaluates reflection loss over a fixed X-band frequency grid.

Primary metrics:

- `RL_min`: minimum reflection loss over the band
- `EAB_10`: maximum continuous bandwidth where `RL <= -10 dB`

Auxiliary engineering proxies:

- effective density
- cost proxy

The final scalar objective is:

`combined_score = reward(EAB_10, |RL_min|) - penalty(thickness, density, cost)`

All ranges and weights are defined in `references/problem_config.json`. The evaluator implementation in `verification/evaluator.py` is the ground truth.

## 4. Output Contract

The candidate must write `temp/submission.json` with:

```json
{
  "benchmark_id": "microwave_absorber_single_layer_xband",
  "d_mm": 2.5,
  "phi_dielectric": 0.20,
  "phi_magnetic": 0.35,
  "phi_matrix": 0.45
}
```

## 5. Validity Rules

A submission is invalid if:

- the JSON file is missing or malformed
- required keys are absent
- `benchmark_id` mismatches
- any value is non-finite or out of range
- fractions do not sum to 1.0 within tolerance
- the candidate times out or exits non-zero
