# DC Optimal Power Flow — IEEE 14-Bus System

## Overview

Minimize generation cost in a 14-bus power network while satisfying the
DC power flow equations and thermal line limits. This is the canonical
**DC Optimal Power Flow (DC-OPF)** problem on the IEEE 14-bus test system.

| Property | Value |
|----------|-------|
| Domain | Power Systems |
| Task form | Continuous optimization with physics constraints |
| Difficulty | 3 / 5 — requires modelling DC power flow; line 2-3 is binding |
| Dependencies | `numpy`, `scipy` |
| Human-best cost | **7 892.76 $/h** (DC-OPF optimal via SLSQP) |
| Baseline cost | 9 154.77 $/h (equal dispatch) → score **0.862** |

## Source

- **MATPOWER** — IEEE 14-bus test case (`case14`):
  <https://github.com/MATPOWER/matpower> (>1 000 ★, BSD-3-Clause)
  The IEEE 14-bus system represents a portion of the American Electric Power
  (AEP) system from the early 1960s. Bus data, generator ratings, and
  quadratic cost coefficients are taken directly from MATPOWER `case14`.

- **PGLib-OPF** — power-grid benchmark library:
  <https://github.com/power-grid-lib/pglib-opf> (>500 ★, CC-BY-4.0)
  Line thermal limits (rate_a, MVA) are sourced from the PGLib-OPF
  `pglib_opf_case14_ieee` variant, which augments MATPOWER case14 with
  conductor ratings consistent with the 69/138 kV voltage levels.

## Problem Description

Given a 14-bus, 5-generator, 20-branch power network, find the active
power dispatch **Pg = [Pg₁, Pg₂, Pg₃, Pg₄, Pg₅]** (MW) that minimises
total generation cost:

```
cost = Σᵢ (aᵢ · Pgᵢ² + bᵢ · Pgᵢ)   [$/h]
```

Subject to:

1. **Power balance**: `sum(Pg) = 259 MW` (total system load)
2. **Generator limits**: `Pmin_i ≤ Pg_i ≤ Pmax_i`
3. **DC line flow limits**: `|Pᵢⱼ| ≤ rate_aᵢⱼ`  for every branch
   (flows computed via DC power flow: `B · θ = P_net`)

The line thermal limit on branch 2→3 (36 MW) is **binding** at optimum,
making the problem non-trivial — naive merit-order dispatch violates this
constraint.

## Task

Write `solve(instance) -> list[float]` in `baseline/init.py`.

```python
def solve(instance: dict) -> list[float]:
    """
    Returns generator dispatch [Pg1, Pg2, Pg3, Pg4, Pg5] in MW.
    """
```

The `instance` dict contains full network data (buses, generators,
branches with reactances and thermal limits).

## Scoring

```
score = min(1.0, HUMAN_BEST_COST / solution_cost)
      = min(1.0, 7892.76 / cost)
```

Higher is better; 1.0 = matches or beats the DC-OPF optimal.
Invalid solutions (power imbalance, generation limit violation, or
line thermal limit violation) receive score = 0.

## How to Run

```bash
cd benchmarks/PowerSystems/OptimalPowerFlow
python verification/evaluate.py
```

## Generator Data

| Gen | Bus | Pmin (MW) | Pmax (MW) | Cost: a ($/MW²/h) | Cost: b ($/MWh) |
|-----|-----|-----------|-----------|-------------------|-----------------|
| 1   | 1   | 0         | 332.4     | 0.0430            | 20              |
| 2   | 2   | 0         | 140.0     | 0.2500            | 20              |
| 3   | 3   | 0         | 100.0     | 0.0100            | 40              |
| 4   | 6   | 0         | 100.0     | 0.0100            | 40              |
| 5   | 8   | 0         | 100.0     | 0.0100            | 40              |
