# Evaluation Summary (reference-time-limit = 2s)

| Family | Instances | Ref Fail | Avg Best Score (Base) | Avg Best Score (Ref) | Delta (Ref-Base) | Avg LB Score (Base) | Avg LB Score (Ref) | Delta (Ref-Base) | Avg Gap% (Base) | Avg Gap% (Ref) | Delta Gap% (Ref-Base) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ft | 3 | 0 | 80.35 | 100.00 | 19.65 | 80.35 | 100.00 | 19.65 | 28.08 | 0.00 | -28.08 |
| la | 40 | 0 | 83.94 | 99.70 | 15.76 | 83.94 | 99.70 | 15.76 | 19.96 | 0.31 | -19.65 |
| abz | 5 | 0 | 80.50 | 96.68 | 16.18 | 80.07 | 96.12 | 16.05 | 21.53 | 2.81 | -18.72 |
| orb | 10 | 0 | 79.45 | 100.00 | 20.55 | 79.45 | 100.00 | 20.55 | 26.30 | 0.00 | -26.30 |
| swv | 20 | 0 | 81.63 | 89.85 | 8.22 | 80.94 | 89.04 | 8.10 | 19.66 | 11.01 | -8.65 |
| yn | 4 | 0 | 76.88 | 92.89 | 16.01 | 74.65 | 90.24 | 15.59 | 35.29 | 6.00 | -29.29 |
| ta | 80 | 1 | 78.80 | 91.11 | 12.31 | 78.33 | 90.56 | 12.23 | 25.82 | 8.84 | -16.98 |

## Weighted overall (by instance count)

- Total instances: 162
- Avg best-known score: baseline=80.49, reference=94.00, delta=13.51
- Avg lower-bound score: baseline=80.11, reference=93.55, delta=13.44
- Avg optimality gap%: baseline=23.79, reference=6.04, delta=-17.75

## Notes

- Raw outputs are stored as `*.txt` per family in the same folder.
- Reference solver was run with 2s time limit per instance.
