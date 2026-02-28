# Frontier Eval Framework

Evaluation framework for `Frontier-Engineering`.

## Layout

- `frontier_eval/cli.py`: main entrypoint (`python -m frontier_eval`)
- `frontier_eval/tasks/`: all evaluation tasks
- `frontier_eval/algorithms/`: all algorithms (currently supports `abmcts`, `openevolve`, `shinkaevolve`)
- `frontier_eval/conf/`: Hydra configs (`task` / `algorithm` / `llm`)

## Setup

Conda is recommended.

The simplest way is to run from the repo root:

```bash
bash init.sh
conda activate frontier-eval
```

Manual setup:

```bash
conda create -n frontier-eval python=3.12 -y
conda activate frontier-eval

# Octave + signal/control
conda install -c conda-forge octave octave-signal octave-control -y

pip install -r frontier_eval/requirements.txt
```

Optional (ShinkaEvolve):

```bash
# NOTE: the PyPI package `shinka` is NOT ShinkaEvolve.
# Recommended (editable VCS install so `shinka.core` is available):
pip install -e "git+https://github.com/SakanaAI/ShinkaEvolve.git#egg=shinka"
```

Optional (AB-MCTS via TreeQuest):

```bash
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

## Extending (new task / algorithm)

- Recommended for most new benchmarks: use `task=unified` + benchmark-local metadata files (section above), no new Python task code needed.
- New custom task (only when unified is insufficient): implement a `frontier_eval/tasks/base.py` `Task` subclass (`initial_program_path()` + `evaluate_program()`), and register it in `frontier_eval/registry_tasks.py` (or keep using `frontier_eval/registry.py`'s `get_task`).
- New algorithm: implement a `frontier_eval/algorithms/base.py` `Algorithm` subclass, and register it in `frontier_eval/registry_algorithms.py`.
