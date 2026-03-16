"""
SABR Calibration — Baseline Solver
===================================
Estimate alpha from the ATM implied vol, assume zero skew (rho=0)
and moderate vol-of-vol (nu=0.3). This ignores the volatility smile
and smirk present in real option markets.
"""
from __future__ import annotations

import math
from typing import Any


# ======================== EVOLVE-BLOCK-START ========================
def solve(instance: dict[str, Any]) -> list[float]:
    """Return SABR parameters [alpha, rho, nu] (beta is fixed at 0.5).

    The SABR model (Hagan 2002) describes the implied volatility smile:
        sigma_SABR(F, K, T; alpha, rho, nu) ≈ ...
    Goal: minimize RMSE between model vols and market_vols over all
    (strike, expiry) pairs provided in the instance.

    Args:
        instance: dict with keys:
            F          - ATM forward price
            beta       - SABR beta (fixed at 0.5)
            T_list     - list of expiries [years]
            strikes    - list of strike prices
            market_vols - 2-D list [n_expiry x n_strikes] of implied vols

    Returns:
        [alpha, rho, nu] - SABR parameters
            alpha > 0 : vol level (≈ ATM_vol * F^(1-beta))
            -1 < rho < 1 : correlation (negative = put skew)
            nu > 0    : vol of vol (governs smile curvature)
    """
    F = instance["F"]
    beta = instance["beta"]
    T_list = instance["T_list"]
    strikes = instance["strikes"]
    market_vols = instance["market_vols"]

    # Use ATM vol from longest expiry to estimate alpha
    # ATM SABR vol ≈ alpha / F^(1-beta)  =>  alpha ≈ ATM_vol * F^(1-beta)
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - F))
    atm_vol = market_vols[-1][atm_idx]   # longest expiry ATM vol
    alpha = atm_vol * (F ** (1.0 - beta))

    # Naive: no skew, moderate vol-of-vol
    rho = 0.0
    nu = 0.3

    return [alpha, rho, nu]
# ======================== EVOLVE-BLOCK-END ========================


if __name__ == "__main__":
    from verification.evaluate import INSTANCE, compute_rmse, evaluate
    params = solve(INSTANCE)
    alpha, rho, nu = params
    F, beta = INSTANCE["F"], INSTANCE["beta"]
    T_list, strikes, mv = INSTANCE["T_list"], INSTANCE["strikes"], INSTANCE["market_vols"]
    rmse = compute_rmse(F, beta, T_list, strikes, mv, alpha, rho, nu)
    print(f"Baseline: alpha={alpha:.4f}, rho={rho:.4f}, nu={nu:.4f}")
    print(f"RMSE = {rmse:.5f}")
