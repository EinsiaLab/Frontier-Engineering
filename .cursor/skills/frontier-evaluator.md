---
name: frontier-evaluator
description: Run and debug Frontier-Engineering evaluations with benchmark-specific runtime settings.
user_invocable: true
---

When asked to run or debug evaluation:

1. Read `frontier_eval/README.md` and benchmark README instructions first.
2. Discover env docs with:
   - `python .claude/skills/scripts/discover_env_docs.py <Domain>`
   - `python .claude/skills/scripts/discover_env_docs.py <Domain>/<Task>`
   - `python .claude/skills/scripts/discover_env_docs.py --matrix frontier_eval/conf/batch/example_matrix.yaml`
3. Keep driver env and benchmark runtime env separated.
4. Start with:
   - `python -m frontier_eval task=unified task.benchmark=<Domain>/<Task> algorithm=openevolve algorithm.iterations=0`
   - `python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml`
5. Report exact commands, overrides, and unresolved prerequisites.
