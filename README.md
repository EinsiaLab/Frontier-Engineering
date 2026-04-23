# Frontier-Eng

English | [简体中文](README_zh-CN.md)

[![Homepage](https://img.shields.io/badge/Homepage-lab.einsia.ai-0969DA?style=flat-square&logo=homepage&logoColor=white)](https://lab.einsia.ai/frontier-eng/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.12290-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2604.12290)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hxeVhZNN)

Frontier-Eng is a benchmark for **generative optimization**: agents iteratively edit runnable engineering code, get feedback from frozen verifiers, and improve under a fixed interaction budget. The paper's central claim is simple: engineering performance is usually about the **best design you can discover within budget**, not average pass rate. Frontier-Eng therefore focuses on:

- continuous reward signals instead of binary grading
- real verifiers and simulators instead of judge models
- improvement trajectories instead of single-shot answers

The benchmark currently covers **47 tasks** across computing, quantum information, operations research, robotics and control, optics and communications, and physical sciences. The project homepage and paper frame it as a missing evaluation axis between pass/fail coding benchmarks and real engineering work: most engineering problems start from a feasible baseline and reward iterative improvement, not one-shot correctness.


## 0. Host Requirements

The repository-owned Python environments are automated, but a full `v1` baseline sweep still assumes a Linux host with the following capabilities already in place:

| Requirement | Needed for | Notes |
|---|---|---|
| Standard build tools and outbound network access | all setup paths | required for `uv`, Python wheels, and benchmark-local downloads |
| NVIDIA GPU + working CUDA stack | `KernelEngineering/*`, `Aerodynamics/CarAerodynamicsSensing`, selected robotics tasks | not needed for CPU-only tasks |
| Docker | `EngDesign` | needed only for that task family |
| Octave | `Astrodynamics/MannedLunarLanding` | host-level tool, not installed by `uv` |
| Extra datasets / checkpoints / third-party repos | selected tasks such as `SustainableDataCenterControl`, `CarAerodynamicsSensing`, `MolecularMechanics` | see task README files for exact paths |

If you want the task-by-task preparation details, including host-tool installation notes, read [`run.md`](run.md) before launching the full `v1` problem set.

## Quickstart

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Create the driver environment

From the repository root:

```bash
bash init.sh
source .venvs/frontier-eval-driver/bin/activate
```

This creates the **driver** environment used to run `python -m frontier_eval`.

### 3. Run a smoke test

```bash
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

This does not need an API key and is the fastest way to confirm the framework itself is wired correctly.

## Running Real Benchmarks

There are two layers of environments in this repository:

- `frontier-eval-driver`: the driver environment
- `.venvs/<runtime-name>`: task runtime environments such as `frontier-v1-main`, `frontier-v1-kernel`, and `frontier-v1-summit`

To create the v1 task runtime environments used by the released `v1` problem set:

```bash
bash scripts/env/setup_v1_task_envs.sh
```

The runtime selector now uses:

- `task.runtime.env_name=<name>` to prepend `.venvs/<name>/bin` to `PATH`
- `task.runtime.python_path=uv-env:<name>` when a task must call a runtime interpreter directly

### Single-task baseline run

No LLM key is needed for baseline-only validation:

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=WirelessChannelSimulation/HighReliableSimulation \
  algorithm=openevolve \
  algorithm.iterations=0
```

### Full `v1` baseline sweep

To validate the released `v1` problem set without any model API calls:

```bash
bash scripts/batch/validate_v1_task_envs.sh
```

That command runs the batch config for the `v1` problem set with `algorithm.iterations=0`, which evaluates each task's shipped baseline instead of asking an LLM to improve it.

If you want the full `v1` problem set with normal optimization runs later, see [`run.md`](run.md).

## Where To Go Next

- Framework commands and task onboarding: [`frontier_eval/README.md`](frontier_eval/README.md)
- Batch-running the released `v1` problem set: [`run.md`](run.md)
- Full task list: [`TASK_DETAILS.md`](TASK_DETAILS.md)
- Archived best solutions from published runs: [`baseline_archive/README.md`](baseline_archive/README.md)

## Leaderboard

Detailed leaderboard: [lab.einsia.ai/frontier-eng/leaderboard.html](https://lab.einsia.ai/frontier-eng/leaderboard.html)

| Rank | Model | Average Rank |
| :--: | :--- | --: |
| 1 | Claude Opus 4.6 | 3.18 |
| 2 | GLM-5 | 4.02 |
| 3 | DeepSeek V3.2 | 4.41 |
| 4 | Gemini 3.1 Pro Preview | 5.34 |
| 5 | Grok 4.20 | 5.60 |
| 6 | SEED 2.0 Pro | 5.63 |
| 7 | GPT-5.4 | 5.68 |
| 8 | Qwen3 Coder Next | 6.68 |

## Contributing

Contribution guidelines live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

```bibtex
@article{chi2026frontier,
  title={Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization},
  author={Chi, Yizhe and Hong, Deyao and Jiang, Dapeng and Luo, Tianwei and Yang, Kaisen and Zhang, Boshi and Cao, Zhe and Fan, Xiaoyan and He, Bingxiang and Hao, Han and others},
  journal={arXiv preprint arXiv:2604.12290},
  year={2026}
}
```
