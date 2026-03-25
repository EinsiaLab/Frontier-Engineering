---
name: frontier-evaluator
description: Run, debug, or prepare Frontier-Engineering benchmark evaluations. Use when setting up benchmark runtime environments, reading benchmark README instructions, running `python -m frontier_eval` or `python -m frontier_eval.batch`, checking runtime overrides, or reproducing evaluation failures in this repository.
---

# Frontier Evaluator

Use this skill to help a user run or debug evaluations in `Frontier-Engineering`.

## Core Rule

The framework environment and the benchmark runtime are not the same thing. Always read the relevant benchmark README files before running a benchmark.

## Workflow

1. Resolve the requested scope.
- Single benchmark or task
- Batch matrix
- Smoke test
- Algorithm regression
- Environment audit or repair

2. Discover the relevant docs before changing anything.
- Read `frontier_eval/README.md`.
- Read `benchmarks/<Domain>/README*.md`.
- Read `benchmarks/<Domain>/<Task>/README*.md` when the task has its own README.
- Use the bundled helper when you need to locate environment-related instructions quickly:
```bash
python scripts/discover_env_docs.py <Domain>
python scripts/discover_env_docs.py <Domain>/<Task>
python scripts/discover_env_docs.py --matrix frontier_eval/conf/batch/example_matrix.yaml
```

3. Separate driver and runtime environments.
- Driver environment: the Python used to launch `python -m frontier_eval` or `python -m frontier_eval.batch`.
- Runtime environment: the interpreter or conda env used by the benchmark evaluator.
- If the benchmark docs specify `task.runtime.conda_env`, `task.runtime.python_path`, `task.runtime.use_conda_run=false`, Docker mode, or `third_party/` prerequisites, treat those instructions as the source of truth.

4. Install dependencies with minimal adaptation.
- Prefer `conda run -n <env> python -m pip install ...` over relying on shell activation state.
- Reuse an existing environment when it already matches the benchmark instructions.
- Clone required `third_party/` repositories before declaring setup complete.
- If the benchmark is Docker-first, validate in the documented Docker mode rather than silently switching to local mode.

5. Start with the lightest valid run.
- Prefer `algorithm.iterations=0` or the benchmark's documented dry-run option.
- For `shinkaevolve`, use `algorithm.max_generations=0` when that is the cheapest compatibility check.
- Use `task=smoke` only for framework sanity, not as proof that a benchmark runtime is configured.

6. Preserve runtime overrides exactly.
- Keep documented runtime overrides in the command line instead of assuming framework defaults.
- If a matrix uses portable markers such as `task.runtime.python_path=conda-env:<env-name>`, keep them unchanged.
- When debugging, inspect `.hydra/overrides.yaml` inside the run directory if you need to confirm which runtime override actually landed.

7. Escalate only after the cheap path passes.
- Single benchmark example:
```bash
python -m frontier_eval task=unified task.benchmark=<Domain>/<Task> algorithm=openevolve algorithm.iterations=0
```
- Batch example:
```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml
```
- Framework smoke example:
```bash
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

8. Report the outcome clearly.
- List which environments were created or reused.
- List which commands were run.
- Call out unresolved prerequisites such as missing Docker, missing `third_party/` repos, or unavailable GPUs.
- When a run fails, point to the exact command, override, and missing dependency rather than giving a vague summary.

## Guardrails

- Do not claim a benchmark is configured without reading its README instructions.
- Do not strip benchmark-specific runtime overrides from the final command.
- Do not hardcode machine-local absolute paths into repository files.
- Do not replace a documented Docker workflow with a local fallback unless the user explicitly asks for that tradeoff.

