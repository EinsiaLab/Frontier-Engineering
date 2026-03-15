# TA Re-run Comparison (2s vs 5s reference limit)

| Metric | 2s run | 5s run | Delta (5s-2s) |
|---|---:|---:|---:|
| instances | 80 | 80 | 0 |
| reference failures | 1 | 0 | -1 |
| avg reference runtime (s) | 2.8327 | 5.6468 | 2.8141 |
| avg best-known score (reference) | 91.11 | 93.13 | 2.02 |
| avg lower-bound score (reference) | 90.56 | 92.58 | 2.02 |
| avg optimality gap % (reference) | 8.84 | 6.71 | -2.13 |
| wall elapsed (s) | 235 | 458 | 223 |

## Interpretation

- 5s run removed the single reference failure (ta76 in 2s run).
- Reference solution quality improved (higher scores, lower optimality gap).
- Runtime increased as expected due to larger per-instance time budget.
