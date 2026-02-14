# Frontier Eval Framework

Evaluation framework for `Frontier-Engineering`.

## Layout

- `frontier_eval/cli.py`: main entrypoint (`python -m frontier_eval`)
- `frontier_eval/tasks/`: all evaluation tasks
- `frontier_eval/algorithms/`: all algorithms (currently supports `openevolve`)
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

## Batch runs

Use the batch runner (writes an isolated `run.output_dir` for each combination and aggregates into `summary.jsonl`):

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml
```

## Extending (new task / algorithm)

- New task: implement a `frontier_eval/tasks/base.py` `Task` subclass (`initial_program_path()` + `evaluate_program()`), and register it in `frontier_eval/registry_tasks.py` (or keep using `frontier_eval/registry.py`'s `get_task`).
- New algorithm: implement a `frontier_eval/algorithms/base.py` `Algorithm` subclass, and register it in `frontier_eval/registry_algorithms.py`.
