# LA16 Neighborhood Move Selection Task

## Objective

Guide an adjacent-swap local search on the canonical LA16 Lawrence 10x10 job shop instance.

The benchmark uses one frozen canonical instance: `la16`.
The known optimum for this instance is `945`.

## Submission Contract

Submit one Python file.

For dispatch-rule tasks, define:

```python
def score_operation(operation, state):
    ...
```

For neighborhood-move tasks, define:

```python
def score_move(move, state):
    ...
```

You may optionally define:

```python
MAX_ITERATIONS = 50
```

## Evaluation

Dispatch-rule tasks:

1. Start from an empty schedule.
2. Repeatedly gather the next unscheduled operation from each job.
3. Among operations with the earliest feasible start time, choose the one with highest `score_operation`.
4. Build a complete feasible schedule and compute makespan.

Neighborhood-move tasks:

1. Start from the baseline SPT dispatch schedule.
2. Repeatedly generate adjacent machine-order swap moves.
3. Rank moves by `score_move`.
4. Apply the first improving move in ranked order.
5. Stop when no improving move exists or `MAX_ITERATIONS` is reached.

## Metrics

- `combined_score`: `-candidate_makespan`
- `valid`: `1.0` only when a complete feasible schedule is produced
- `candidate_makespan`
- `baseline_makespan`
- `relative_gap_to_optimum`

## Failure Cases

The submission is marked invalid and receives a very low score if:

- the required scoring function is missing
- the return value is non-finite
- the induced schedule is infeasible
- the candidate crashes during evaluation
