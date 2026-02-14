# Frontier Eval Framework

Evaluation framework for `Frontier-Engineering`.

## Structure

- `frontier_eval/cli.py`: Main entry point (`python -m frontier_eval`)
- `frontier_eval/tasks/`: All evaluation tasks
- `frontier_eval/algorithms/`: All algorithms (currently supports integrating `openevolve`)
- `frontier_eval/conf/`: Hydra configs (task / algorithm / llm)

## Environment Setup

Conda is recommended.

The simplest way is to run this at the repository root:

  bash init.sh
  conda activate frontier-eval

Manual installation:

  conda create -n frontier-eval python=3.12 -y
  conda activate frontier-eval

  # Octave + signal/control
  conda install -c conda-forge octave octave-signal octave-control -y

  pip install -r frontier_eval/requirements.txt

Environment variables (recommended via `.env`):

  cp .env.example .env
  # Edit .env and fill in OPENAI_API_KEY / OPENAI_API_BASE, etc.

When running `python -m frontier_eval ...`, it will automatically search upward from the current directory and load the nearest `.env`.

## Run

  python -m frontier_eval algorithm.iterations=10

## Batch Evaluation

Use the batch runner (it will write each combination into an independent `run.output_dir`, and aggregate results into `summary.jsonl`):

  python -m frontier_eval.batch --matrix frontier_eval/conf/batch/example_matrix.yaml

## Extending (Adding a Task / Algorithm)

- Add a task: implement a `Task` subclass in `frontier_eval/tasks/base.py`
  (initial_program_path() + evaluate_program()), and register it in
  `frontier_eval/registry_tasks.py` (or keep using `get_task` in `frontier_eval/registry.py`).

- Add an algorithm: implement an `Algorithm` subclass in `frontier_eval/algorithms/base.py`,
  and register it in `frontier_eval/registry_algorithms.py`.
