# ReactionOptimisation Benchmark Tasks

This directory packages four SUMMIT reaction-optimization benchmarks as standalone tasks and as `frontier_eval` unified benchmarks.

All commands below are repository-relative and reproducible without any absolute paths.

## Tasks

| Task Folder | Core Objective | Notes |
|---|---|---|
| `snar_multiobjective` | maximize `sty` while minimizing `e_factor` | continuous-flow SnAr Pareto optimization |
| `mit_case1_mixed` | maximize `y` | mixed continuous + categorical catalyst search |
| `reizman_suzuki_pareto` | maximize `yld` while minimizing `ton` | catalyst/process co-design Pareto task |
| `dtlz2_pareto` | approximate the DTLZ2 Pareto front | synthetic multi-objective reference task |

## Folder Layout

Each task directory contains:

- `Task.md`: full problem definition in English
- `Task_zh-CN.md`: full problem definition in Chinese
- `README.md`: short background and folder guide
- `README_zh-CN.md`: short background and folder guide in Chinese
- `task.py`: benchmark construction, sampling helpers, and scoring logic
- `baseline/solution.py`: baseline solver and unified-task editable file
- `verification/reference.py`: stronger SUMMIT-based reference implementation
- `verification/evaluate.py`: evaluates a candidate solver against the reference
- `frontier_eval/`: unified-task metadata consumed by `python -m frontier_eval`

## Environment

Verified setup:

- `.venvs/frontier-v1-summit` for direct verification and benchmark runtime
- `.venvs/frontier-eval-driver` for `python -m frontier_eval`

Example setup from repository root:

```bash
bash init.sh
RUN_VALIDATION=0 bash scripts/env/setup_v1_task_envs.sh
source .venvs/frontier-eval-driver/bin/activate
```

The released matrix uses `frontier-v1-summit` as a dedicated runtime because these tasks are slower and more fragile than the default driver env.

## Direct Verification

Template:

```bash
.venvs/frontier-v1-summit/bin/python benchmarks/ReactionOptimisation/<task>/verification/evaluate.py
```

Verified commands:

```bash
.venvs/frontier-v1-summit/bin/python benchmarks/ReactionOptimisation/snar_multiobjective/verification/evaluate.py
.venvs/frontier-v1-summit/bin/python benchmarks/ReactionOptimisation/mit_case1_mixed/verification/evaluate.py
.venvs/frontier-v1-summit/bin/python benchmarks/ReactionOptimisation/reizman_suzuki_pareto/verification/evaluate.py
.venvs/frontier-v1-summit/bin/python benchmarks/ReactionOptimisation/dtlz2_pareto/verification/evaluate.py
```

Measured runtime on the verified setup:

| Task | Direct Verification Runtime | Notes |
|---|---|---|
| `snar_multiobjective` | ~`122s` | weighted SUMMIT reference over multiple scalarizations |
| `mit_case1_mixed` | ~`106s` | fastest task in this suite |
| `reizman_suzuki_pareto` | ~`112s` | includes catalyst screening plus fixed-catalyst SOBO runs |
| `dtlz2_pareto` | ~`160s` | slowest direct verifier in this suite |

`verification/evaluate.py` runs both the baseline and the reference solver over the default 3 seeds, so it is intentionally heavier than a smoke test.

## frontier_eval (Unified)

All four tasks are integrated through `task=unified` metadata under `benchmarks/ReactionOptimisation/<task>/frontier_eval`.

Template:

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=ReactionOptimisation/<task> \
  task.runtime.python_path=uv-env:frontier-v1-summit \
  algorithm=openevolve \
  algorithm.iterations=0
```

Copy-paste example:

```bash
python -m frontier_eval task=unified task.benchmark=ReactionOptimisation/snar_multiobjective task.runtime.python_path=uv-env:frontier-v1-summit algorithm=openevolve algorithm.iterations=0
```

Verified benchmark IDs:

| Task | `task.benchmark` | `algorithm.iterations=0` Runtime | Notes |
|---|---|---|---|
| `snar_multiobjective` | `ReactionOptimisation/snar_multiobjective` | ~`137s` | full score pipeline still runs at iteration 0 |
| `mit_case1_mixed` | `ReactionOptimisation/mit_case1_mixed` | ~`61s` | shortest unified run |
| `reizman_suzuki_pareto` | `ReactionOptimisation/reizman_suzuki_pareto` | ~`130s` | slower because the evaluator still computes the reference |
| `dtlz2_pareto` | `ReactionOptimisation/dtlz2_pareto` | ~`107s` | under the default `300s` timeout |

Notes:

- `algorithm.iterations=0` is a framework compatibility run, but it still executes one full benchmark evaluation of `baseline/solution.py`.
- The released `v1` matrix uses `algorithm.oe.evaluator.timeout=600` for `snar_multiobjective`, `mit_case1_mixed`, and `reizman_suzuki_pareto`. On slower CPUs, use the same setting.
- If you prefer not to use the shorthand, replace `task.runtime.python_path=uv-env:frontier-v1-summit` with the absolute interpreter path `.venvs/frontier-v1-summit/bin/python`.
- You can confirm the override landed correctly by checking `.hydra/overrides.yaml` in the run directory for `task.runtime.python_path=uv-env:frontier-v1-summit`.

## Current Baselines and References

- `snar_multiobjective`
  baseline: adaptive random scalarization with local mutation around the current scalarized incumbent
  reference: split-budget SUMMIT `SOBO` with multiple scalarization weights
- `mit_case1_mixed`
  baseline: mixed random plus local search
  reference: direct SUMMIT `SOBO` on the mixed domain
- `reizman_suzuki_pareto`
  baseline: pure random search over catalyst and continuous conditions
  reference: catalyst screening followed by fixed-catalyst SUMMIT `SOBO`
- `dtlz2_pareto`
  baseline: random scalarization with local mutation
  reference: split-budget SUMMIT `SOBO` over several scalarization weights
