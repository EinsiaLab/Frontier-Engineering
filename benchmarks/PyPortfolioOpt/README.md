# PyPortfolioOpt

This benchmark family focuses on practical portfolio rebalancing and allocation optimization.
It is built on problem settings and solver workflows that depend on
[PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt).

## Dependency

Install dependencies from repository root:

```bash
pip install -r benchmarks/PyPortfolioOpt/requirements.txt
pip install PyPortfolioOpt
```

## Shared Structure Across Subtasks

- All three tasks optimize portfolio decisions under realistic constraints (exposure, turnover, budget, execution friction).
- All three use deterministic evaluators with seeded instances.
- All three score candidate solutions against a reference optimizer (theoretical upper bound score is `100`).

## Key Differences

- `robust_mvo_rebalance`: robust mean-variance objective with sector/factor/turnover constraints (convex program).
- `cvar_stress_control`: scenario-based tail-risk minimization under return and exposure constraints (CVaR optimization).
- `discrete_rebalance_mip`: integer lot rebalancing with transaction fees and turnover notional limits (mixed-integer program).

## Difficulty (Practical)

- `robust_mvo_rebalance`: Medium
- `cvar_stress_control`: Medium-High
- `discrete_rebalance_mip`: High (integer optimization; higher runtime variance)

## Subtask Index and Unified Run Commands

- `robust_mvo_rebalance/`
  - unified run:
    ```bash
    .venvs/frontier-eval-driver/bin/python -m frontier_eval \
      task=unified \
      task.benchmark=PyPortfolioOpt/robust_mvo_rebalance \
      task.runtime.env_name=frontier-v1-main \
      algorithm.iterations=0
    ```
  - typical `iterations=0` runtime: around 8-15 seconds.

- `cvar_stress_control/`
  - unified run:
    ```bash
    .venvs/frontier-eval-driver/bin/python -m frontier_eval \
      task=unified \
      task.benchmark=PyPortfolioOpt/cvar_stress_control \
      task.runtime.env_name=frontier-v1-main \
      algorithm.iterations=0
    ```
  - typical `iterations=0` runtime: around 9-18 seconds.

- `discrete_rebalance_mip/`
  - unified run:
    ```bash
    .venvs/frontier-eval-driver/bin/python -m frontier_eval \
      task=unified \
      task.benchmark=PyPortfolioOpt/discrete_rebalance_mip \
      task.runtime.env_name=frontier-v1-main \
      algorithm.iterations=0
    ```
  - typical `iterations=0` runtime: around 8-20 seconds, sometimes longer on slower CPUs.

For longer evolutionary runs (`algorithm.iterations > 0`), total runtime grows approximately linearly with iteration count.
