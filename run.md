# Frontier-Eng: running the benchmark

Chinese: [run_zh-CN.md](run_zh-CN.md)

Framework-level commands live in [`frontier_eval/README.md`](frontier_eval/README.md). This page focuses on the released `v1` problem set and the operator workflow around it.

## 0. Host requirements and what this repo automates

This repository automates the Python environments it owns. It does not install host-level tools, GPU drivers, Docker, or large third-party benchmark assets for you.

For a full `v1` baseline sweep, assume you need:

| Requirement | Needed for | How to verify |
|---|---|---|
| Linux shell environment with standard build tools and outbound network access | all setup paths | `python3 --version`, `git --version`, `curl --version` |
| NVIDIA GPU plus working CUDA runtime | `KernelEngineering/*`, `Aerodynamics/CarAerodynamicsSensing`, selected robotics tasks | `nvidia-smi` and a successful CUDA PyTorch import |
| Docker | `EngDesign` | `docker version` |
| Octave | `Astrodynamics/MannedLunarLanding` | `octave --version` |
| Task-local assets / checkpoints / third-party repos | selected tasks | check the task README for exact files and paths |

### Install host tools when needed

Examples below assume Ubuntu or another Debian-like Linux distribution.

If you want the repository to drive the common apt-based setup for you, use:

```bash
bash scripts/bootstrap/install_host_deps.sh --octave
bash scripts/bootstrap/install_host_deps.sh --docker --configure-docker-group
```

#### Octave

```bash
sudo apt-get update
sudo apt-get install -y octave
octave --version
```

#### Docker

If your distro packages are sufficient for local evaluation:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER"
docker version
```

Start a new shell after changing group membership. If you need the full upstream setup instead, follow the official Docker Engine guide: [docs.docker.com/engine/install](https://docs.docker.com/engine/install/).

#### CUDA / NVIDIA stack

This repository does not install GPU drivers or CUDA for you. Before running GPU-required tasks, verify:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

If those checks fail, install a working NVIDIA driver / CUDA stack first, following your cluster or machine's standard setup procedure.

### External assets that are still manual

The repo now includes an asset bootstrap helper for benchmarks and optional algorithm repos:

```bash
python scripts/bootstrap/fetch_task_assets.py --list
python scripts/bootstrap/fetch_task_assets.py --target v1-baseline-assets
python scripts/bootstrap/fetch_task_assets.py --target shinkaevolve
python scripts/bootstrap/fetch_task_assets.py --target abmcts
```

`v1-baseline-assets` covers the currently automated benchmark-side bundles, such as PhySense assets for `Aerodynamics/CarAerodynamicsSensing` and the upstream `dc-rl` checkout path for `SustainableDataCenterControl` when that vendored tree is absent.

| Requirement | Affected tasks | Where to look |
|---|---|---|
| `dc-rl` checkout and SustainDC assets | `SustainableDataCenterControl` | `benchmarks/SustainableDataCenterControl/README.md` and `hand_written_control/README.md` |
| PhySense dataset, checkpoints, and reference points | `Aerodynamics/CarAerodynamicsSensing` | `benchmarks/Aerodynamics/CarAerodynamicsSensing/README.md` |
| `openff-dev` runtime | `MolecularMechanics/*` | `bash scripts/bootstrap/install_openff_dev.sh`, then see `benchmarks/MolecularMechanics/README.md` |

## 1. Prepare the environments

From the repo root:

```bash
bash init.sh
bash scripts/env/setup_v1_task_envs.sh
source .venvs/frontier-eval-driver/bin/activate
```

That gives you:

- `.venvs/frontier-eval-driver` for the driver
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

## 3. Run the released `v1` problem set

### Standard batch run

```bash
bash scripts/batch/run_v1_batch.sh
```

This launches:

```bash
python -m frontier_eval.batch --matrix frontier_eval/conf/batch/v1.yaml
```

through the driver interpreter in `.venvs/frontier-eval-driver`.

Useful variants:

```bash
bash scripts/batch/run_v1_batch.sh --dry-run
bash scripts/batch/run_v1_batch.sh --tasks KernelEngineering/MLA
bash scripts/batch/run_v1_batch.sh --exclude-tasks engdesign
```

### Baseline-only validation

To verify the shipped baselines without any LLM calls:

```bash
bash scripts/batch/validate_v1_task_envs.sh
```

This runs the batch config for the released `v1` problem set with `algorithm.iterations=0` and splits validation into CPU, GPU, and kernel subsets.

Important: this command validates the tasks only after their host prerequisites and external assets are already in place. It is not a promise that a fresh machine with only `uv` installed will automatically pass every task.

## 4. Important runtime knobs

- `CUDA_VISIBLE_DEVICES`: select the GPU for GPU-heavy tasks
- `GPU_DEVICES`: GPU id used by `scripts/batch/validate_v1_task_envs.sh`
- `DRIVER_ENV`: defaults to `frontier-eval-driver`
- `DRIVER_PY`: explicit path to the driver Python if you do not want to use the default `.venvs/frontier-eval-driver/bin/python`
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
