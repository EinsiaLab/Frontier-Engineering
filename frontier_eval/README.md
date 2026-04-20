# Frontier Eval Framework

Evaluation framework for `Frontier-Engineering`.

## Layout

- `frontier_eval/cli.py`: main evaluation entrypoint (`python -m frontier_eval`)
- `frontier_eval/tasks/`: all evaluation tasks
- `frontier_eval/algorithms/`: all algorithms (currently supports `abmcts`, `openevolve`, `shinkaevolve`)
- `frontier_eval/conf/`: Hydra configs (`task` / `algorithm` / `llm`)

## Setup

Conda is recommended.

The simplest way is to run from the repo root:

```bash
bash init.sh
conda activate frontier-eval-2
```

Manual setup (only if you cannot use `init.sh`):

```bash
conda create -n frontier-eval-2 python=3.12 -y
conda activate frontier-eval-2

# Octave + signal/control
conda install -c conda-forge octave octave-signal octave-control -y

pip install -r frontier_eval/requirements.txt
```

Important:

The setup above only prepares the `frontier_eval` framework. Many benchmarks in this repository need a separate runtime environment, extra benchmark-local requirements, `third_party/` repos, or Docker.

Before running a specific benchmark, always read:

1. `benchmarks/<Domain>/README*.md`
2. `benchmarks/<Domain>/<Task>/README*.md` when that task has its own README

Treat those benchmark README files as the source of truth for runtime setup and copy the documented overrides into your command, such as `task.runtime.conda_env=...`, `task.runtime.python_path=...`, or `task.runtime.use_conda_run=false`.

Common examples in this repository:

- `ReactionOptimisation` uses `summit` for benchmark runtime.
- `MolecularMechanics` uses `openff-dev`.
- `SustainableDataCenterControl` uses `sustaindc`.
- `PyPortfolioOpt` uses `pyportfolioopt`.
- `QuantumComputing` uses `mqt`.
- `InventoryOptimization` uses `stock`.
- `JobShop` uses an explicit `task.runtime.python_path`.
- `EngDesign` uses Docker-first execution or local fallback.

Note on `third_party/`:

Some optional algorithms/benchmarks depend on extra repos under `third_party/`. In this repo, `third_party/` is meant for local checkouts and is ignored by git (see `.gitignore`), so if you see commands like `pip install -e third_party/...`, clone the corresponding repo first, e.g.:

```bash
mkdir -p third_party

# AB-MCTS / TreeQuest (required if you run `algorithm=abmcts`)
git clone https://github.com/SakanaAI/treequest.git third_party/treequest

# CarAerodynamicsSensing / PhySense (required to evaluate that benchmark)
git clone https://github.com/thuml/PhySense.git third_party/PhySense
```

Optional (ShinkaEvolve):

```bash
# NOTE: the PyPI package `shinka` is NOT ShinkaEvolve.

# Option A: local checkout under `third_party/` (recommended if you need to apply local patches)
git clone https://github.com/SakanaAI/ShinkaEvolve.git third_party/ShinkaEvolve
# Frontier-Engineering patch: fixes `DatabaseDisplay` when `program.metadata` is missing,
# and adds the OpenRouter model id `qwen/qwen3-coder-next` to the pricing table.
git apply patches/third_party_shinkaevolve.patch
pip install -e third_party/ShinkaEvolve

# Option B: editable VCS install so `shinka.core` is available:
pip install -e "git+https://github.com/SakanaAI/ShinkaEvolve.git#egg=shinka"
```

Using your own model with `algorithm=shinkaevolve`:

- If you point `OPENAI_API_BASE` / `llm.api_base` at a self-hosted or OpenAI-compatible endpoint, ShinkaEvolve still resolves model routing and cost metadata from `third_party/ShinkaEvolve/shinka/llm/providers/pricing.csv`.
- If `OPENAI_MODEL` / `llm.model` is missing from that table, ShinkaEvolve may fail with errors such as `Model ... not supported.` or `Model ... not found in pricing data`.
- If you need to patch `pricing.csv`, `client.py`, or `query.py`, use Option A above (`third_party/ShinkaEvolve` checkout), not the VCS install.
- For a custom model on an OpenAI-compatible endpoint, add a row to `third_party/ShinkaEvolve/shinka/llm/providers/pricing.csv` and usually set `provider` to `openai`, because the `openai` backend is the one that honors `OPENAI_API_BASE`.

Example `pricing.csv` row for a self-hosted OpenAI-compatible model:

```csv
my-model,openai,N/A,N/A,,,," False"," 0"," 0"
```

Then set the same model name in your runtime config:

```bash
export OPENAI_API_BASE=http://<your-endpoint>/v1
export OPENAI_API_KEY=<your-key>
export OPENAI_MODEL=my-model
```

If your model needs a new provider or a non-standard API, editing `pricing.csv` is not enough. Follow `third_party/ShinkaEvolve/docs/support_local_llm.md` and update `shinka/llm/client.py` plus `shinka/llm/query.py`.

Optional (AB-MCTS via TreeQuest):

```bash
# Requires the TreeQuest repo in `third_party/treequest` (see above).
pip install -e third_party/treequest
# Optional (ABMCTS-M / mixed model):
pip install -e "third_party/treequest[abmcts-m]"
# Optional (tree visualization):
pip install -e "third_party/treequest[vis]"
```

Environment variables (recommended: `.env`):

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY / OPENAI_API_BASE etc.
```

When running `python -m frontier_eval ...`, it will automatically search upwards from the current directory and load the nearest `.env`.

## Run

```bash
python -m frontier_eval algorithm.iterations=10
```

Quick smoke (fast, no external benchmark deps):

```bash
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
python -m frontier_eval task=smoke algorithm=shinkaevolve algorithm.max_generations=0
python -m frontier_eval task=smoke algorithm=abmcts algorithm.iterations=0
```

## Unified task

Use `task=unified` to onboard a new benchmark by adding metadata files under the benchmark folder, instead of implementing a new `frontier_eval/tasks/<task>/...`.

Run example:

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=ComputerSystems/MallocLab \
  algorithm=openevolve \
  algorithm.iterations=10
```

EngDesign example (preset config, still unified under the hood):

```bash
python -m frontier_eval \
  task=engdesign \
  algorithm=openevolve \
  algorithm.iterations=10
```

Equivalent explicit unified command:

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=EngDesign \
  task.runtime.use_conda_run=false \
  algorithm=openevolve \
  algorithm.iterations=10
```

EngDesign runtime mode (from `benchmarks/EngDesign/README.md`):
- `benchmarks/EngDesign/frontier_eval/run_eval.sh` uses `docker` first when available (`ENGDESIGN_EVAL_MODE=auto`).
- Set `ENGDESIGN_EVAL_MODE=local` to force local Python evaluation.

When `task=unified`, default run directory includes benchmark id:
- `runs/unified__<Domain>__<Task>/<algorithm>/<model>/<timestamp>`

#### Benchmark metadata layout

Under `benchmarks/<Domain>/<Task>/frontier_eval/`:

```text
initial_program.txt      # required: relative path to baseline candidate file
candidate_destination.txt# optional: where candidate is copied in sandbox
eval_command.txt         # required: benchmark eval command template
eval_cwd.txt             # optional: working dir (relative to benchmark root)
agent_files.txt          # optional: files exposed to agent as artifacts
copy_files.txt           # optional: files/dirs copied into temp sandbox
readonly_files.txt       # optional: files/dirs that must stay unchanged
artifact_files.txt       # optional: files/dirs auto-collected by framework
constraints.txt          # optional: prompt/constraints text for agent
```

Line-based `*.txt` files (`initial_program.txt`, `candidate_destination.txt`, `eval_cwd.txt`, `agent_files.txt`, `copy_files.txt`, `readonly_files.txt`, `artifact_files.txt`) support:
- one path per line
- empty lines ignored
- lines starting with `#` ignored

`eval_command.txt` is raw shell command text (can be multi-line).

#### What each metadata file means

- `initial_program.txt`: initial source file used by evolution (relative to benchmark root).
- `candidate_destination.txt`: path in sandbox benchmark where each candidate is written. If omitted, defaults to `initial_program.txt`.
- `eval_command.txt`: evaluator command template.
- `eval_cwd.txt`: command working directory in sandbox. `.` means benchmark root.
- `agent_files.txt`: files/dirs loaded into artifacts for LLM context.
- `copy_files.txt`: files/dirs copied into sandbox. If empty, unified copies the entire benchmark directory.
- `readonly_files.txt`: files/dirs fingerprinted before/after eval. Any change marks run invalid.
- `artifact_files.txt`: files/dirs collected by unified framework after eval (for example logs/output files). This avoids writing custom artifacts-export code.
- `constraints.txt`: free-form instruction text attached to artifacts (agent prompt context).

#### Placeholder reference

Safe placeholders (shell-escaped, recommended):
- `{python}`: runtime python command.
- `{candidate}`: candidate file path in sandbox.
- `{benchmark}`: sandbox benchmark root.
- `{sandbox}`: sandbox temp root for this evaluation.
- `{repo_root}`: Frontier-Engineering repo root.
- `{benchmark_source}`: original benchmark directory on disk.
- `{benchmark_id}`: normalized benchmark id (for example `ComputerSystems/MallocLab`).

Raw placeholders (not shell-escaped):
- `{python_raw}`, `{candidate_raw}`, `{benchmark_raw}`, `{sandbox_raw}`, `{repo_root_raw}`, `{benchmark_source_raw}`, `{benchmark_id_raw}`.
- Use raw placeholders only when you explicitly handle quoting in your command.

Example:

```text
bash frontier_eval/run_eval.sh {python} {benchmark} {candidate}
```

Default outputs expected from your eval command:
- `metrics.json`: a JSON object. Unified reads all numeric-like fields (int/float/bool/numeric string), not only `combined_score` and `valid`.
- `artifacts.json` (optional): a JSON object with extra structured artifacts.
- For simple tasks, you can skip `artifacts.json` and use `artifact_files.txt` so unified collects logs automatically.
- If `valid` is missing, unified falls back to command return code (`0 -> 1`, non-zero -> `0`).
- If `combined_score` is missing, unified falls back to `1` when `valid > 0`, else `0`.
- If eval command returns non-zero, unified forces `valid=0` and `combined_score=0`.

If your output paths differ, override at runtime:

```bash
python -m frontier_eval task=unified \
  task.benchmark=MyDomain/MyTask \
  task.metrics_json=verification/out/metrics.json \
  task.artifacts_json=verification/out/artifacts.json
```


For specific examples, please refer to `benchmarks/ComputerSystems/MallocLab/frontier_eval`

### Environment selection

`unified` supports passing benchmark runtime environment:
- default conda env: `frontier-eval-2`
- override env name: `task.runtime.conda_env=<env_name>`
- pass explicit Python path: `task.runtime.python_path=/path/to/python`

Important:

- The default `frontier-eval-2` is only a fallback for unified runs.
- Many benchmarks in this repository should not use that default as their actual runtime.
- Always check the benchmark README first and prefer its documented env name / Python path / Docker mode over the generic default shown here.

Example:

```bash
python -m frontier_eval task=unified \
  task.benchmark=MyDomain/MyTask \
  task.runtime.conda_env=frontier-eval-2
```

## Batch runs

Use the batch runner (writes an isolated `run.output_dir` for each combination and aggregates into `summary.jsonl`):

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml
```

### v1 unified batch matrix

**v1** batch runs use **`frontier_eval/conf/batch/v1.yaml`**. `OPENAI_API_BASE`, `OPENAI_MODEL`, and related settings are read from the environment when the matrix loads (same conventions as `frontier_eval/conf/llm/openai_compatible.yaml`).

For host setup (Docker env vars for EngDesign, `CUDA_VISIBLE_DEVICES`, merged conda envs, etc.), see **[`run_en.md`](../run_en.md)** · [`run.md`](../run.md) at the repository root. Run the v1 batch matrix with **`bash scripts/run_v1_batch.sh`** (forwards extra args to `frontier_eval.batch`).

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

Use `--tasks` / `--exclude-tasks`, or lower `run.max_parallel` in the YAML, to avoid contention when mixing CPU and GPU workloads.

`matrix.tasks` supports either plain task names or labeled task entries with per-entry overrides:

```yaml
tasks:
  - manned_lunar_landing
  - name: unified
    label: ReactionOptimisation/dtlz2_pareto
    overrides:
      - task.benchmark=ReactionOptimisation/dtlz2_pareto
      - task.runtime.conda_env=summit
```

The `label` is used for filtering (`--tasks` / `--exclude-tasks`), run directory names,
and `summary.jsonl` rows, while `name` remains the actual Hydra task config.

Rerun a subset of tasks:

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml \
  --tasks denoising --tasks trimul
```

Rerun in-place inside an existing batch directory (deletes the selected task directories first):

```bash
python -m frontier_eval.batch --matrix runs/batch/<batch_id>/matrix_resolved.yaml \
  --in-place --tasks denoising
```

Unified baseline sweep example:

```bash
.venv/bin/python -m frontier_eval.batch \
  --matrix frontier_eval/conf/batch/shinkaevolve_unified_baselines.yaml \
  --python .venv/bin/python
```

## Extending (new task / algorithm)

- Required default for new benchmark contributions: use `task=unified` + benchmark-local metadata files (section above). New benchmark PRs should onboard through the unified format instead of adding new Python task code under `frontier_eval/tasks/`.
- New custom task (only when unified is insufficient and the exception has been discussed with maintainers): implement a `frontier_eval/tasks/base.py` `Task` subclass (`initial_program_path()` + `evaluate_program()`), and register it in `frontier_eval/registry_tasks.py` (or keep using `frontier_eval/registry.py`'s `get_task`).
- New algorithm: implement a `frontier_eval/algorithms/base.py` `Algorithm` subclass, and register it in `frontier_eval/registry_algorithms.py`.

## v1 Task Environments

To reduce the number of runtime environments used by the effective `v1` task pool, the repository now provides declarative env specs under `scripts/env_specs/` and one-shot setup scripts:

- `frontier-eval-2` is managed from `scripts/env_specs/frontier-eval-2.yml` as the default driver env.
- Merged task environments are managed from repo-owned manifests (`frontier-v1-main`, `frontier-v1-summit`, `frontier-v1-sustaindc`, `frontier-v1-kernel`) instead of cloning local ad-hoc envs.
- For `v1` tasks that need a direct interpreter instead of `conda run` (currently `ReactionOptimisation/*` and `JobShop/*`), the batch matrices use the portable marker `conda-env:<env-name>`. The unified evaluator resolves that marker to the target env's Python executable at runtime, so repository files stay machine-independent.

Current `v1` runtime consolidation:

- `frontier-v1-main`: `SingleCellAnalysis/predict_modality`, `QuantumComputing/*`, `Optics/*`, `InventoryOptimization/*`, `PyPortfolioOpt/*`, `JobShop/*`, `Robotics/DynamicObstacleAvoidanceNavigation`, `Robotics/PIDTuning`, `Robotics/UAVInspectionCoverageWithWind`, `Robotics/QuadrupedGaitOptimization`, `Robotics/RobotArmCycleTimeOptimization`, `Aerodynamics/CarAerodynamicsSensing`, `KernelEngineering/FlashAttention`
- `frontier-v1-summit`: `ReactionOptimisation/*`
- `frontier-v1-sustaindc`: `SustainableDataCenterControl/*`
- `frontier-v1-kernel`: `KernelEngineering/MLA`, `KernelEngineering/TriMul`

If an older benchmark README still mentions legacy env names such as `mqt`, `stock`, `pyportfolioopt`, or `jobshop`, prefer **`frontier_eval/conf/batch/v1.yaml`** (and [`run_en.md`](../run_en.md) / [`run.md`](../run.md) for operator setup) as the source of truth for current `v1` runs.

Setup and validation scripts:

- Initialize/update merged envs from declarative specs (and run post-build `iter=0` validation by default): `bash scripts/setup_v1_merged_task_envs.sh`
- Validate merged envs with `iter=0`: `DRIVER_ENV=frontier-eval-2 GPU_DEVICES=<gpu_id> bash scripts/validate_v1_merged_task_envs.sh`
- Audit benchmark readonly metadata coverage: `python scripts/audit_unified_metadata_readonly.py [--strict]`

Notes:

- The validation script uses `conda run -n frontier-eval-2 python` as the default driver, and can also be overridden with `DRIVER_PY=/path/to/python`. It checks CPU `v1`, GPU `v1`, and a kernel batch (`MLA`, `TriMul`, `FlashAttention`) from `v1.yaml`.
- `MuonTomography` is listed in [TASK_DETAILS.md](../TASK_DETAILS.md) but is **not** included in [`frontier_eval/conf/batch/v1.yaml`](../frontier_eval/conf/batch/v1.yaml).
- Known caveat: the official `KernelEngineering/TriMul` full benchmark (`verification/tri_bench.txt`) may still be VRAM-limited on 24GB-class GPUs; this is typically a task-level memory-bound issue rather than a missing dependency in `frontier-v1-kernel`.
