---
name: frontier-contributor
description: Help contribute new or updated benchmarks to Frontier-Engineering. Use when adding a benchmark, onboarding a task through `task=unified`, implementing `scripts/init.py` and `verification/evaluator.py`, validating local benchmark runs, or preparing a clean pull request for this repository.
---

# Frontier Contributor

Use this skill to help a user contribute benchmark content to `Frontier-Engineering`.

## Principles

- Prefer the unified benchmark path.
- Keep contributions runnable, reproducible, and easy to review.
- Optimize for a feasible initial solution, a trustworthy evaluator, and clean local validation.
- Avoid secrets, absolute paths, machine-local assumptions, and stray temporary files.

## Workflow

1. Read the scope before editing.
- Inspect `README.md` and `frontier_eval/README.md`.
- For an existing benchmark, also read `benchmarks/<Domain>/README*.md` and `benchmarks/<Domain>/<Task>/README*.md`.
- For a new benchmark, inspect a similar benchmark in the same domain before deciding the file layout.

2. Check the repository bar.
- Reality gap: keep the task close to a real engineering problem rather than a pure toy exercise.
- Economic or engineering value: improving the objective should matter in practice.
- Verifiability: provide an automatic evaluator that finishes in acceptable time.

3. Prefer unified onboarding.
- Default to benchmark-local metadata under `<Task>/frontier_eval/` and validate with `task=unified`.
- Only add new Python task code under `frontier_eval/tasks/` when the unified format is clearly insufficient and the maintainers have agreed to that exception.

4. Implement the editable candidate program.
- Put the user-improvable entry point in `scripts/init.py` unless the benchmark already uses a different evolved file.
- Keep important logic inside the editable file so optimization algorithms have enough context.
- Keep the evolved file self-contained relative to the benchmark. Do not depend on helper modules from other benchmark files unless the benchmark contract requires it.
- Preserve CLI, I/O, and evaluator contracts outside the editable region.
- Add `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers when the benchmark is meant for ShinkaEvolve or ABMCTS style editing, or whenever a single editable region should be enforced.

5. Implement verification.
- Add `verification/evaluator.py` as the scoring entry point.
- Add `verification/requirements.txt` for benchmark-specific evaluator dependencies.
- Add Docker support only when the benchmark genuinely needs it.
- Make the evaluator produce stable machine-readable outputs and fail loudly on invalid candidates.

6. Keep repository hygiene high.
- Remove `.env`, credentials, IDE settings, local caches, logs, `__pycache__`, and personal scripts.
- Use relative paths only.
- Do not add unnecessary generated documentation.
- If the user explicitly asks for AI-generated documentation such as `README.md` or `Task.md`, append `<!-- AI_GENERATED -->` at the end of each generated document.

7. Run mandatory checks before considering the work done.
- Benchmark-level check:
```bash
python verification/evaluator.py scripts/init.py
```
- Framework integration check for new or unified benchmarks:
```bash
python -m frontier_eval task=unified task.benchmark=<Domain>/<Task> algorithm=openevolve algorithm.iterations=0
```
- If the benchmark docs require runtime overrides such as `task.runtime.conda_env=...`, `task.runtime.python_path=...`, or Docker-only execution, keep those overrides exactly in the validation command.
- If the benchmark is a legacy custom task rather than `task=unified`, run the registered task name instead of forcing unified.

8. Prepare a reviewable pull request.
- Summarize the task background, engineering value, and source.
- Include the exact local validation commands and their outputs.
- State any remaining runtime prerequisites such as Docker images or `third_party/` clones.

## Guardrails

- Do not assume `frontier_eval/requirements.txt` is enough for a benchmark runtime.
- Do not silently replace benchmark-specific runtime instructions with generic defaults.
- Do not split critical evolved logic into hidden helper modules.
- Do not leave benchmark submissions half-validated.

## Contribution Checklist

- [ ] Benchmark concept meets the reality-gap, value, and verifiability bar.
- [ ] Benchmark metadata is onboarded through `task=unified`, unless maintainers approved an exception.
- [ ] `scripts/init.py` is feasible and runnable.
- [ ] `verification/evaluator.py` and `verification/requirements.txt` exist and work.
- [ ] Required runtime overrides are documented and preserved in test commands.
- [ ] Local tests passed with real command output captured for the PR.

