# Bridge Topology Optimization Task

## Objective

Minimize compliance on a frozen bridge-like topology optimization case with a passive-solid deck and distributed load.

The benchmark freezes one pyMOTO-based structural optimization case in `runtime/problem.py`.

## Submission Contract

Submit one Python file that defines:

```python
def update_density(density, sensitivity, state):
    ...
```

Inputs:

- `density`: current density vector as a NumPy array of shape `(nel,)`
- `sensitivity`: current compliance sensitivity with respect to the design vector
- `state`: a dict containing:
  - `iteration`
  - `domain_shape`
  - `volume_fraction`
  - `target_density_sum`
  - `minimum_density`
  - `move_limit`
  - `current_compliance`
  - `history`
  - `passive_solid_mask`
  - `passive_void_mask`

The function must return the next feasible density vector. A dict with key `density` is also accepted.

You may import `project_density` from `runtime.problem` if you want a helper that projects a raw proposal back onto the feasible set.

## Evaluation

The evaluator will:

1. Build the frozen pyMOTO finite-element model.
2. Run 30 fixed optimization iterations.
3. Compare the baseline OC update rule against your `update_density(...)`.
4. Reject non-finite or infeasible density updates.
5. Expose the final candidate compliance directly as the optimization score.

## Metrics

- `combined_score`: `-candidate_compliance`
- `valid`: `1.0` only if every density update is finite and feasible
- `candidate_compliance`
- `baseline_compliance`
- `final_volume_fraction`
- `volume_fraction_error`

## Failure Cases

The submission is marked invalid and receives a very low score if:

- `update_density()` is missing
- any proposed density is non-finite
- any density violates bounds, move limits, passive masks, or volume budget
- the pyMOTO solve fails
