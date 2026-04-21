# Frontier-Eng: running the benchmark

Chinese: [run_zh-CN.md](run_zh-CN.md)

Framework-level commands live in [`frontier_eval/README.md`](frontier_eval/README.md). This page focuses on the released `v1` matrix and the operator workflow around it.

## 1. Prepare the environments

From the repo root:

```bash
bash init.sh
bash scripts/setup_v1_merged_task_envs.sh
source .venvs/frontier-eval-2/bin/activate
```

That gives you:

- `.venvs/frontier-eval-2` for the driver
- `.venvs/frontier-v1-main` for most CPU tasks
- `.venvs/frontier-v1-summit` for `ReactionOptimisation/*`
- `.venvs/frontier-v1-sustaindc` for `SustainableDataCenterControl/*`
- `.venvs/frontier-v1-kernel` for kernel/GPU runtimes

Before longer runs:

```bash
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
```

## 2. Configure model access

Optimization runs need a working `.env`:

```bash
cp .env.example .env
```

Set at least:

- `OPENAI_API_KEY`
- optionally `OPENAI_API_BASE`
- optionally `OPENAI_MODEL`

Baseline-only validation does **not** need an API key as long as you run with `algorithm.iterations=0`.

## 3. Run the released `v1` matrix

### Standard batch run

```bash
bash scripts/run_v1_batch.sh
```

This launches:

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

through the driver interpreter in `.venvs/frontier-eval-2`.

Useful variants:

```bash
bash scripts/run_v1_batch.sh --dry-run
bash scripts/run_v1_batch.sh --tasks KernelEngineering/MLA
bash scripts/run_v1_batch.sh --exclude-tasks engdesign
```

### Baseline-only validation

To verify the shipped baselines without any LLM calls:

```bash
bash scripts/validate_v1_merged_task_envs.sh
```

This runs the same matrix with `algorithm.iterations=0` and splits validation into CPU, GPU, and kernel subsets.

## 4. Important runtime knobs

- `CUDA_VISIBLE_DEVICES`: select the GPU for GPU-heavy tasks
- `GPU_DEVICES`: GPU id used by `scripts/validate_v1_merged_task_envs.sh`
- `DRIVER_ENV`: defaults to `frontier-eval-2`
- `DRIVER_PY`: explicit path to the driver Python if you do not want to use the default `.venvs/frontier-eval-2/bin/python`
- `V1_MATRIX`: override the matrix path
- `ENGDESIGN_EVAL_MODE`, `ENGDESIGN_DOCKER_IMAGE`: see [`benchmarks/EngDesign/README.md`](benchmarks/EngDesign/README.md)

## 5. What a successful baseline sweep does and does not prove

A baseline-only run is valuable because it verifies:

- the Hydra config resolves correctly
- the benchmark runtime starts
- the evaluator can execute the shipped baseline
- `metrics.json` / `artifacts.json` handling is wired correctly

It does **not** prove that every benchmark is fully self-contained on a fresh machine. Some tasks still require external assets, Docker, Octave, CUDA, or benchmark-local data before the baseline can run successfully.

## 6. Output locations

Batch results are written under:

```text
runs/batch/<run.name>/
```

Validation runs use:

```text
runs/batch_validation/
```

Each task gets its own output directory, and aggregated summaries are stored in `summary.jsonl`.
