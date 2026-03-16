# Task: SABR Volatility Surface Calibration

## Task Name

`sabr_calibration`

## Task Description

Implement a `solve(instance)` function that returns SABR model parameters `[alpha, rho, nu]`
minimising the RMSE between model-implied vols and market-observed vols over 33 (strike, expiry)
combinations.

The SABR model (Hagan 2002) closed-form approximation for implied volatility:

For F ≠ K:
```
σ(F,K,T) = α / { (FK)^((1-β)/2) · [1 + (1-β)²/24·ln²(F/K) + (1-β)⁴/1920·ln⁴(F/K)] }
           × z/χ(z) × [1 + ( (1-β)²α² / (24(FK)^(1-β))
                              + ρβνα / (4(FK)^((1-β)/2))
                              + (2-3ρ²)ν²/24 ) · T ]

where z = ν/α · (FK)^((1-β)/2) · ln(F/K)
      χ(z) = ln[ (√(1-2ρz+z²) + z - ρ) / (1-ρ) ]
```

For F = K (ATM):
```
σ_ATM = α/F^(1-β) · [1 + ((1-β)²α²/(24F^(2-2β)) + ρβνα/(4F^(1-β)) + (2-3ρ²)ν²/24) · T]
```

## Interface

```python
def solve(instance: dict) -> list[float]:
    """
    Args:
        instance: {
            'F': float,           # ATM forward price (100.0)
            'beta': float,        # Fixed at 0.5
            'T_list': list,       # [0.25, 0.50, 1.00] years
            'strikes': list,      # [70,...,120] (11 values)
            'market_vols': list,  # shape [3][11] implied vols
        }
    Returns:
        [alpha, rho, nu]  with  alpha>0, -1<rho<1, nu>0
    """
```

## Scoring

```
RMSE = sqrt( mean( (sabr_vol(F, K, T, α, ρ, ν) - market_vol[T,K])² ) )
combined_score = min(1.0, 0.001 / RMSE)
```

Higher score is better. Score = 1.0 when RMSE ≤ 0.001.

## Baseline Score

Baseline (α from ATM vol, ρ=0, ν=0.3): score ≈ **0.046**

## Human Best Score

Properly calibrated SABR: score = **1.000** (RMSE < 1e-6)

## Constraints

- `alpha > 0`
- `-1 < rho < 1`
- `nu > 0`
- `beta` is fixed at 0.5 (do not change it)
