# HighReliableSimulation Task

## Objective

Estimate BER for Hamming(127,120) over AWGN in a rare-event regime.
You must implement `MySampler` and support variance-controlled simulation.

## Submission Contract

Submit one Python file that defines:

1. `class MySampler(SamplerBase)`
2. `MySampler.simulate_variance_controlled(...)`

Evaluator call pattern:

```python
sampler = MySampler(code=code, seed=seed)
result = sampler.simulate_variance_controlled(
    code=code,
    sigma=DEV_SIGMA,
    target_std=TARGET_STD,
    max_samples=MAX_SAMPLES,
    batch_size=BATCH_SIZE,
    fix_tx=True,
    min_errors=MIN_ERRORS,
)
```

`code` is fixed by evaluator as `HammingCode(r=7, decoder="binary")` with `ChaseDecoder(t=3)`.

## Return Format

Accepted formats:

- Tuple/list with at least 6 fields:
  `(errors_log, weights_log, err_ratio, total_samples, actual_std, converged)`
- Dict with equivalent keys.

`err_rate_log` is interpreted as `errors_log - weights_log`.

## Frozen Evaluation Constants

- `sigma = 0.268`
- `target_std = 0.05`
- `max_samples = 100000`
- `batch_size = 10000`
- `min_errors = 20`
- `r0 = 5.52431776694918e-07`
- `t0 = 0.18551087379455566`
- `epsilon = 0.8`
- `repeats = 3`

## Scoring

- `e = |log(r / r0)|`, where `r = exp(err_rate_log)`.
- If `e >= epsilon`, score is `0`.
- Otherwise: `score = t0 / (t * e + 1e-6)`, where `t` is median runtime.

## Failure Cases

Score is `0` if any of the following happens:

- missing or invalid `MySampler` interface
- invalid return value or non-finite metrics
- runtime failure
