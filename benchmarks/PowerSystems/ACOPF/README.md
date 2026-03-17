# AC Optimal Power Flow — PGLib-OPF Style (5-bus DC Subset)

Minimize total generation cost subject to DC power flow constraints and generation/voltage limits. This benchmark uses a minimal 5-bus case so the full pipeline runs without external MATPOWER data.

## Source

| Item | Detail |
|------|--------|
| **Benchmark** | [power-grid-lib/pglib-opf](https://github.com/power-grid-lib/pglib-opf) — 388+ ★, CC-BY-4.0 data |
| **Formulation** | IEEE PES standard AC-OPF; this task uses a 5-bus DC-OPF subset for self-contained evaluation |

## Background

AC Optimal Power Flow is the core economic dispatch problem: minimize generation cost while satisfying power flow equations and operational limits. The full AC formulation is non-convex; this benchmark uses the DC approximation (linear power flow) on a 5-bus network so that evaluator and baseline can run in pure Python.

## Difficulty

**3 / 5** — DC-OPF is convex (QP), but requires building the B matrix and handling bounds; extending to AC-OPF is non-trivial.

## Scoring

```
combined_score = min(1.0, HUMAN_BEST_COST / total_generation_cost)
HUMAN_BEST_COST = 26.0   # optimal for embedded 5-bus case
```

- Minimization: lower cost → higher score.
- Valid solution must satisfy power balance and generation limits.

## Human Best

Reference optimal cost for the embedded 5-bus case: **26.0**. Baseline (DC-OPF with SLSQP) reaches score 1.0; worse dispatches yield score < 1.0.
