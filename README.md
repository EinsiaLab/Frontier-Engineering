<div align="center">

# Frontier-Engineering Bench

**Evaluating frontier AI on real engineering optimization — not toy problems.**

[![Tasks](https://img.shields.io/badge/Tasks-95+-blue?style=flat-square)](benchmarks/)
[![Domains](https://img.shields.io/badge/Domains-24-purple?style=flat-square)](benchmarks/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Live-brightgreen?style=flat-square)](https://einsia.github.io/Einsia-lab/frontier-eng/)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-orange?style=flat-square)](CONTRIBUTING.md)

English | [简体中文](README_zh-CN.md)

</div>

---

## What is Frontier-Eng?

Unlike benchmarks that reduce engineering to math puzzles, Frontier-Eng puts AI agents on problems with **real physical constraints and economic stakes** — structural loads, chemical yield, orbital mechanics, GPU kernel throughput. You either find a better solution, or you don't.

Three things make it different:

- **Open-ended scoring** — continuous improvement matters, not just pass/fail
- **Peak performance focus** — we care about the best a model can achieve, not its average
- **Real-world complexity** — problems keep the domain knowledge, constraints, and tooling

## Task Coverage

| Domain | Tasks | Domain | Tasks |
|--------|------:|--------|------:|
| 🔬 Optics | 17 | 🏭 Job Shop Scheduling | 10 |
| 🏗️ Engineering Design | 9 | 🤖 Robotics | 6 |
| ⚗️ Reaction Optimisation | 5 | 📦 Inventory Optimization | 5 |
| 🧬 Molecular Mechanics | 4 | 🏛️ Structural Optimization | 4 |
| 🔬 Single Cell Analysis | 4 | 📡 Communication Eng. | 3 |
| 🔐 Cryptographic | 3 | 🔌 EDA | 3 |
| ⚙️ Kernel Engineering | 3 | 📈 Portfolio Optimization | 3 |
| ⚛️ Quantum Computing | 3 | ✈️ Aerodynamics | 2 |
| 💻 Computer Systems | 2 | ⚡ Energy Storage | 2 |
| 🌍 Particle Physics | 2 | 🚀 Astrodynamics | 1 |
| 🔩 Additive Manufacturing | 1 | 🔋 Power Systems | 1 |
| 🌿 Sustainable Data Center | 1 | 📶 Wireless Channel Sim. | 1 |

→ Full task list and status: [TASK_PROGRESS.md](TASK_PROGRESS.md)

## Quick Start

```bash
# Run the evaluation framework
conda activate frontier-eval-2
python -m frontier_eval task=unified task.benchmark=Robotics/PIDTuning algorithm.iterations=5

# Install agent skills
python -m frontier_eval skill evaluator codex
```

<details>
<summary>⚙️ Runtime environment notes</summary>

`frontier_eval/requirements.txt` only sets up the evaluation framework itself — not every benchmark runs in the same environment.

**Always read the benchmark-specific README before running** (`benchmarks/<Domain>/README.md` or `benchmarks/<Domain>/<Task>/README.md`). When a README documents overrides like `task.runtime.conda_env=...` or `task.runtime.use_conda_run=false`, those take precedence over defaults.

**v1 consolidated environments:**

| Environment | Tasks |
|---|---|
| `frontier-v1-main` | QuantumComputing, Optics, InventoryOptimization, PyPortfolioOpt, JobShop, most Robotics, Aerodynamics, KernelEngineering/FlashAttention |
| `frontier-v1-summit` | ReactionOptimisation/\* |
| `frontier-v1-sustaindc` | SustainableDataCenterControl/\* |
| `frontier-v1-kernel` | KernelEngineering/MLA, KernelEngineering/TriMul |

Setup: `bash scripts/setup_v1_merged_task_envs.sh`  
Validate: `DRIVER_ENV=frontier-eval-2 GPU_DEVICES=<id> bash scripts/validate_v1_merged_task_envs.sh`

</details>

## Contributing a New Task

We welcome PRs for new engineering problems. Three requirements:

1. **Reality gap** — must reflect real physical/engineering constraints, not pure math
2. **Economic value** — solving it should matter in practice
3. **Verifiable** — provide a runnable `evaluator.py` (Docker preferred)

> AI-assisted contributions are welcome. Run `python -m frontier_eval skill` to install the `Contributor` skill, but keep human review in the loop.

<details>
<summary>📁 Required file structure per task</summary>

```
<Domain>/
├── README.md                    # Domain overview
└── <TaskName>/
    ├── README.md                # How to run, quick start
    ├── Task.md                  # Background, model, I/O definitions
    ├── references/              # constants.json, manuals, etc.
    ├── frontier_eval/           # Unified task metadata
    │   ├── initial_program.txt
    │   ├── eval_command.txt
    │   ├── agent_files.txt
    │   └── artifact_files.txt
    ├── verification/
    │   ├── evaluator.py         # Scoring entry point
    │   ├── requirements.txt
    │   └── docker/Dockerfile
    └── baseline/                # Optional reference solution
```

New tasks must use the unified format (`task=unified`). See `frontier_eval/README.md` for the full metadata schema.

**EVOLVE-BLOCK markers** are required for ShinkaEvolve / ABMCTS — wrap the editable region in the evolved file:
```python
# EVOLVE-BLOCK-START
...your code...
# EVOLVE-BLOCK-END
```

</details>

<details>
<summary>🔀 Contribution process</summary>

1. Fork → create branch `feat/<Domain>/<TaskName>`
2. Add task files following the structure above
3. Test locally: `python verification/evaluator.py scripts/init.py`
4. Validate with framework: `python -m frontier_eval task=unified task.benchmark=<Domain>/<Task> algorithm.iterations=0`
5. Open PR with a brief description of the task background and how to run it
6. AI Agent does an automated first-pass review; maintainers do final merge

> First time? Open an Issue first to discuss your task idea.

</details>

<details>
<summary>📦 best_code_only snapshot</summary>

`best_code_only/` contains the global best code for each experiment/algorithm/model/task combination.

```
best_code_only/
└── <experiment>/<algorithm>/<model>/<task>/<code-file>
```

Check `best_code_only/coverage.json` first to confirm coverage. Example paths:
- `best_code_only/experiment1/openevolve/gpt-5.4/Astrodynamics_MannedLunarLanding/`
- `best_code_only/experiment2/shinkaevolve/claude-opus-4.6/KernelEngineering_TriMul/`

</details>

## Evaluation Framework

Core implementation: [`frontier_eval/`](frontier_eval/) — see the [Evaluation README](frontier_eval/README.md) for full usage. Some algorithms require local clones under `third_party/`.

## Community

| | |
|---|---|
| 🟢 Feishu | [Join discussion group](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) |
| 💬 Discord | [Join community](https://discord.gg/hxeVhZNN) |

<!-- AI_GENERATED -->
