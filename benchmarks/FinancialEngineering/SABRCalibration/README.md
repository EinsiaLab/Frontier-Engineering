# SABR Volatility Surface Calibration

## Overview

Calibrate the SABR stochastic volatility model (Hagan et al. 2002) to a market-observed implied
volatility surface. Given option implied vols at multiple strikes and expiries, find the three
SABR parameters (α, ρ, ν) that best reproduce the market smile.

This task captures the **volatility smile** problem central to quantitative finance: market option
prices imply different Black-Scholes volatilities at different strikes, forming a characteristic
"smile" or "skew" that naïve flat-vol models cannot reproduce.

## Source

| Item | Detail |
|------|--------|
| **Benchmark** | Hagan, Kumar, Lesniewski, Woodward (2002) "Managing Smile Risk", *Willmott Magazine* |
| **Reference implementation** | [QuantLib](https://github.com/lballabio/QuantLib) — C++ quantitative finance library, **>4 000 ★**, LGPL |
| **Market data structure** | Representative SPX option vol surface (normalised forward F=100) |
| **Parameters** | α=2.5, ρ=-0.65, ν=0.40, β=0.5 (β fixed) |

## Problem Description

The SABR model (Hagan 2002) describes stochastic volatility via:

```
dF = α·F^β · dW₁
dα = ν·α · dW₂
corr(dW₁, dW₂) = ρ dt
```

The analytical approximation for implied volatility σ_SABR(F, K, T; α, ρ, ν) is a closed-form
formula (with ATM and non-ATM branches). The calibration task is:

**minimise RMSE** = √( mean{ (σ_SABR(F, Kᵢ, Tⱼ; α,ρ,ν) − σ_market[j,i])² } )

over the 33 (T, K) pairs in the dataset (3 expiries × 11 strikes).

## Difficulty

**3 / 5** — Moderate.

- Requires implementing the SABR closed-form formula (ATM branch handling is non-trivial).
- Non-convex objective with multiple local minima if initialised poorly.
- Simple scaling of α from ATM vol is a necessary insight; skew and curvature require
  proper rho/nu estimation.

## Input / Output

| Field | Type | Description |
|-------|------|-------------|
| `F` | float | Normalised ATM forward (100.0) |
| `beta` | float | Fixed SABR beta = 0.5 |
| `T_list` | list[float] | Expiries [0.25, 0.50, 1.00] years |
| `strikes` | list[int] | [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120] |
| `market_vols` | list[list[float]] | 3×11 market implied vols |

**Output**: `[alpha, rho, nu]` — three floats satisfying `alpha>0`, `-1<rho<1`, `nu>0`.

## Validity

Always valid if output is `[alpha, rho, nu]` with `alpha>0`, `-1<rho<1`, `nu>0`.

## Scoring

```
combined_score = min(1.0, HUMAN_BEST_RMSE / RMSE)
HUMAN_BEST_RMSE = 0.001  (achievable with proper SABR calibration)
```

- Baseline (flat ATM vol, no skew): RMSE ≈ 0.022 → score ≈ **0.046**
- Well-calibrated SABR: RMSE < 0.001 → score = **1.000**

## Human Best / Best-Known

Proper SABR calibration with scipy.optimize recovers the true parameters (α=2.5, ρ=-0.65, ν=0.40)
to machine precision → RMSE < 1e-6 → `combined_score = 1.0`.

Reference: Hagan et al. (2002) equation (2.17b).

## Running

```bash
# Baseline
python baseline/init.py

# Evaluate a candidate
python verification/evaluate.py baseline/init.py
```

## Dependencies

```
numpy
scipy
```

No GPU, no Docker, no external data download required.
