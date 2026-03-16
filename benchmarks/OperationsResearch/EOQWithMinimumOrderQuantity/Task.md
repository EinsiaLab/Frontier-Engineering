# EOQ with Minimum Order Quantity Task

## Objective

Optimize annual cost for deterministic EOQ instances with a hard minimum order quantity.

Canonical source lineage comes from `Stockpyl` EOQ routines and standard deterministic EOQ formulas. The benchmark uses frozen benchmark-local cases defined in `runtime/problem.py`.

## Submission Contract

Submit one Python file that defines:

```python
def solve(instance):
    ...
```

The return value must be:

- For EOQ tasks: a dict with `order_quantity`, or a raw numeric quantity.
- For `(r,Q)` tasks: a dict with `reorder_point` and `order_quantity`, or a 2-tuple `(r, Q)`.

## Evaluation

The evaluator will:

1. Load the frozen case set from `runtime/problem.py`.
2. Run the reference baseline for each case.
3. Run your `solve(instance)` implementation for each case.
4. Convert the returned quantity or `(r, Q)` pair into a cost and feasibility result.
5. Compute the average candidate cost and expose it directly as the optimization score.

## Metrics

- `combined_score`: `-avg_cost`
- `valid`: `1.0` only if every case is feasible and every output is finite
- `avg_cost`: average candidate cost
- `avg_cost_ratio`: average `baseline_cost / candidate_cost` for diagnostics only

## Failure Cases

The submission is marked invalid and receives a very low score if:

- `solve()` is missing
- the returned output cannot be parsed
- any case violates feasibility constraints
- any metric becomes non-finite
