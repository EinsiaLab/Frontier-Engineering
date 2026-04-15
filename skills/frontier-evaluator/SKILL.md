---
name: frontier-evaluator
description: Run and debug Frontier-Engineering evaluations. Use for benchmark runtime setup, command execution, override checks, and failure reproduction.
---

# frontier-evaluator

## When to use
- Run one benchmark or a batch matrix.
- Reproduce or debug evaluation failures.
- Verify runtime overrides from benchmark docs.

## Workflow
1. Read `frontier_eval/README.md` and benchmark README files first.
2. Locate env setup docs quickly with:
   - `python skills/frontier-evaluator/scripts/discover_env_docs.py <Domain>`
   - `python skills/frontier-evaluator/scripts/discover_env_docs.py <Domain>/<Task>`
   - `python skills/frontier-evaluator/scripts/discover_env_docs.py --matrix frontier_eval/conf/batch/example_matrix.yaml`
3. Keep driver env and benchmark runtime env separate.
4. Start with the lightest valid run:
   - `python -m frontier_eval task=unified task.benchmark=<Domain>/<Task> algorithm=openevolve algorithm.iterations=0`
   - `python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml`
5. Report exact commands, overrides, and unresolved prerequisites.

## Guardrails
- Do not skip benchmark README runtime instructions.
- Do not strip benchmark-specific runtime overrides.
- Do not replace documented Docker mode unless explicitly requested.
