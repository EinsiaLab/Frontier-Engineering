---
name: frontier-contributor
description: Add or update Frontier-Engineering benchmarks. Use for benchmark onboarding, scripts/init.py changes, verifier updates, and contribution validation.
---

# frontier-contributor

## When to use
- Add a new benchmark.
- Update an existing benchmark task.
- Prepare contribution checks for a benchmark PR.

## Workflow
1. Read `README.md`, `frontier_eval/README.md`, and benchmark README files.
2. Prefer unified onboarding (`task=unified`) unless maintainers require an exception.
3. Keep editable logic in `scripts/init.py` and evaluator in `verification/evaluator.py`.
4. Keep outputs reproducible; avoid secrets, absolute paths, and local-only files.
5. Run:
   - `python verification/evaluator.py scripts/init.py`
   - `python -m frontier_eval task=unified task.benchmark=<Domain>/<Task> algorithm=openevolve algorithm.iterations=0`

## Guardrails
- Keep benchmark-specific runtime overrides unchanged.
- Do not claim done without real local validation output.
