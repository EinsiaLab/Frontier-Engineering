# Inventory Optimization Tasks

This folder contains 5 related but distinct hard-mode tasks built on stockpyl.

## Environment Setup
```bash
pip install stockpyl numpy scipy
```

## Common Pattern
Each task uses the same structure:
- `baseline/init.py`: simple non-optimizer implementation
- `verification/reference.py`: stockpyl-based reference implementation
- `verification/evaluate.py`: evaluates both methods using one scoring function
- `output/`: result artifacts (`baseline_result.json`, `reference_result.json`, `comparison.json`)

## How Tasks Are Related
- All tasks optimize inventory decisions under uncertainty.
- All tasks use a 0-1 weighted score and compare baseline vs reference.
- All tasks are configured so baseline is intentionally weaker than reference.

## How Tasks Differ
1. `tree_gsm_safety_stock`
   - Problem class: tree-structured multi-echelon safety-stock placement
   - Key model: GSM (committed service times)
2. `general_meio`
   - Problem class: general-topology MEIO with simulation-based objective
   - Key model: base-stock optimization in non-tree network
3. `joint_replenishment`
   - Problem class: multi-SKU joint replenishment
   - Key model: cycle/multiple coordination across SKUs
4. `finite_horizon_dp`
   - Problem class: finite-horizon dynamic ordering
   - Key model: time-varying `(s_t, S_t)` policy
5. `disruption_eoqd`
   - Problem class: lot sizing with supply disruptions
   - Key model: EOQ with disruption risk

## Difficulty Guide
- `tree_gsm_safety_stock`: Medium (2.5/5)
  - Why: small tree and low-dimensional decision space, but requires understanding SLA/complexity constraints in scoring.
- `disruption_eoqd`: Medium+ (3/5)
  - Why: one-dimensional decision variable, but multi-objective tradeoff (cost, service, risk, capital) under stochastic simulation.
- `joint_replenishment`: Medium-High (3.5/5)
  - Why: mixed discrete-continuous optimization (`m_i`, `T`) and nontrivial coordination/responsiveness tradeoff.
- `finite_horizon_dp`: High (4/5)
  - Why: dynamic policy design over time, stochastic demand, and simulation-based metric tuning.
- `general_meio`: Very High (4.5/5)
  - Why: multi-echelon network, simulation-driven objective, and strong robustness/balance requirements across stress scenarios.

## Run All Evaluations
Run this from `benchmarks/InventoryOptimization/`:

```bash
for d in tree_gsm_safety_stock general_meio joint_replenishment finite_horizon_dp disruption_eoqd; do
  echo "== $d =="
  python "$d/verification/evaluate.py"
done
```

## frontier_eval Unified Task Integration
Each subtask now includes benchmark-local unified metadata under:

- `<subtask>/frontier_eval/initial_program.txt`
- `<subtask>/frontier_eval/candidate_destination.txt`
- `<subtask>/frontier_eval/eval_command.txt`
- `<subtask>/frontier_eval/agent_files.txt`
- `<subtask>/frontier_eval/readonly_files.txt`
- `<subtask>/frontier_eval/artifact_files.txt`
- `<subtask>/frontier_eval/constraints.txt`
- `<subtask>/frontier_eval/run_eval.py`

Run unified task from repository root (driver in `.venvs/frontier-eval-driver`, benchmark runtime in `.venvs/frontier-v1-main`):

```bash
.venvs/frontier-eval-driver/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=InventoryOptimization/tree_gsm_safety_stock \
  task.runtime.env_name=frontier-v1-main \
  algorithm=openevolve \
  algorithm.iterations=0
```

Run all 5 subtasks:

```bash
for d in tree_gsm_safety_stock general_meio joint_replenishment finite_horizon_dp disruption_eoqd; do
  .venvs/frontier-eval-driver/bin/python -m frontier_eval \
    task=unified \
    task.benchmark=InventoryOptimization/$d \
    task.runtime.env_name=frontier-v1-main \
    algorithm=openevolve \
    algorithm.iterations=0
done
```

## Runtime Notes
Measured on local verification run (`frontier-v1-main` runtime), one evaluation per subtask:

- `tree_gsm_safety_stock`: about 3-4s
- `general_meio`: about 20-25s (longer task, simulation-heavy)
- `joint_replenishment`: about 1s
- `finite_horizon_dp`: about 6s
- `disruption_eoqd`: about 3s
