---
name: frontier-contributor
description: Add or update Frontier-Engineering benchmarks and prepare contribution checks.
user_invocable: true
---

When asked to contribute or update a benchmark:

1. Read `README.md`, `frontier_eval/README.md`, and related benchmark READMEs.
2. Prefer unified onboarding (`task=unified`) unless maintainers require otherwise.
3. Keep editable logic in `scripts/init.py` and evaluator logic in `verification/evaluator.py`.
4. Run:
   - `python verification/evaluator.py scripts/init.py`
   - `python -m frontier_eval task=unified task.benchmark=<Domain>/<Task> algorithm=openevolve algorithm.iterations=0`
5. Keep runtime overrides unchanged and avoid secrets or machine‑local paths.
