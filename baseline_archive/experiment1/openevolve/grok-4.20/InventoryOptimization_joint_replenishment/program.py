# EVOLVE-BLOCK-START
"""Baseline for Task 03: all m=1 with optimal T."""

from __future__ import annotations


def solve() -> dict:
    """All-multiples-1 policy using closed-form optimal base cycle (simplified)."""
    K, ks, hs, ds = 100.0, [40., 35., 30., 28., 25., 22., 20., 18.], [1.8, 2., 1.6, 1.7, 1.5, 1.9, 2.1, 1.4], [120., 90., 60., 40., 25., 18., 12., 8.]
    A = K + sum(ks)
    B = sum(h * d for h, d in zip(hs, ds)) / 2.
    T = (A / B) ** .5
    ms = [1] * len(ds)
    qs = [d * T for d in ds]
    return {"base_cycle_time": T, "order_multiples": ms, "order_quantities": qs}
# EVOLVE-BLOCK-END
