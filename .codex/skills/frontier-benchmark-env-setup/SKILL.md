---
name: frontier-benchmark-env-setup
description: Configure Frontier-Engineering benchmark environments by reading benchmark README instructions, creating the required driver/runtime envs, installing dependencies, and validating the documented runtime overrides. Use when asked to set up, repair, or audit environments for one or more benchmarks in this repository.
---

# Frontier Benchmark Env Setup

Use this skill when the user wants help preparing environments for `Frontier-Engineering` benchmarks.

The main rule is simple:

- Do not assume the framework env is enough.
- Always read the benchmark README files first.

## Scope

This repo mixes several environment patterns:

- shared framework env for `python -m frontier_eval`
- separate benchmark runtime env selected by `task.runtime.conda_env`
- explicit benchmark runtime interpreter via `task.runtime.python_path`
- Docker-first evaluation for some tasks
- extra `third_party/` checkouts for selected algorithms or benchmarks

Examples already present in this repo:

- `ReactionOptimisation`: driver env `frontier-eval-2`, runtime env `summit`
- `MolecularMechanics`: driver env `frontier-eval-2`, runtime env `openff-dev`
- `SustainableDataCenterControl`: driver env `frontier-eval-2`, runtime env `sustaindc`
- `PyPortfolioOpt`: runtime env `pyportfolioopt`
- `QuantumComputing`: runtime env `mqt`
- `InventoryOptimization`: runtime env `stock`
- `JobShop`: explicit `task.runtime.python_path` plus `task.runtime.use_conda_run=false`
- `EngDesign`: Docker-first runtime

## Workflow

1. Resolve the requested scope.
- If the user names one benchmark like `ReactionOptimisation/snar_multiobjective`, configure that benchmark only.
- If the user points to a batch matrix, discover all referenced benchmarks first.
- If the user asks for "all needed environments", configure only the environments required by the benchmarks in scope, not every env in the repository by default.

2. Discover the right README files before making changes.
- Run:
```bash
python .codex/skills/frontier-benchmark-env-setup/scripts/discover_env_docs.py <benchmark-or-domain>
```
- For batch configs, run:
```bash
python .codex/skills/frontier-benchmark-env-setup/scripts/discover_env_docs.py --matrix <matrix.yaml>
```
- Open the reported `benchmarks/<Domain>/README*.md` and `benchmarks/<Domain>/<Task>/README*.md` files.
- Also consult `frontier_eval/README*.md` for framework-level defaults.

3. Separate driver env from benchmark runtime env.
- Driver env: the Python used to run `python -m frontier_eval`.
- Runtime env: the Python/env used by the benchmark evaluator itself.
- Prefer the benchmark README over generic defaults whenever they differ.

4. Install dependencies using the documented commands with minimal adaptation.
- Prefer `conda run -n <env> python -m pip install ...` over relying on shell activation state.
- Reuse an existing env if it already matches the benchmark README.
- Do not recreate environments blindly if a working env already exists.
- If the README requires Docker or `third_party/`, do that before declaring setup complete.

5. Verify with the lightest documented command.
- Prefer the README's compatibility run, usually `algorithm.iterations=0`.
- If the benchmark offers lighter debug knobs, use them.
- For heavy families like `JobShop`, use the documented runtime env vars to limit the check.

6. Confirm the runtime override really landed.
- For `task.runtime.conda_env=...`, inspect the run directory `.hydra/overrides.yaml` when needed.
- For `task.runtime.python_path=...`, confirm the command matches the documented interpreter.
- For Docker-backed tasks, verify the README's expected mode or env var.

7. Report back clearly.
- List which envs were created or reused.
- List which commands were run.
- Call out any manual prerequisite still missing.

## Guardrails

- Never assume `frontier_eval/requirements.txt` is enough for a benchmark runtime.
- Never strip benchmark-specific runtime overrides from the final run command.
- Do not hardcode absolute paths into repository files.
- If the benchmark README itself still contains a machine-local absolute path, use it only for the live setup session and call it out as a portability issue.

## Handy Commands

Single benchmark:

```bash
python .codex/skills/frontier-benchmark-env-setup/scripts/discover_env_docs.py ReactionOptimisation/snar_multiobjective
```

Domain overview:

```bash
python .codex/skills/frontier-benchmark-env-setup/scripts/discover_env_docs.py MolecularMechanics
```

Batch matrix:

```bash
python .codex/skills/frontier-benchmark-env-setup/scripts/discover_env_docs.py --matrix frontier_eval/conf/batch/example_matrix.yaml
```

Repository-wide environment doc sweep:

```bash
python .codex/skills/frontier-benchmark-env-setup/scripts/discover_env_docs.py --all
```

<!-- AI_GENERATED -->
