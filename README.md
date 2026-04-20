# Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization

English | [简体中文](README_zh-CN.md)

[![Homepage](https://img.shields.io/badge/Homepage-lab.einsia.ai-0969DA?style=flat-square&logo=homepage&logoColor=white)](https://lab.einsia.ai/frontier-eng/) [![arXiv](https://img.shields.io/badge/arXiv-2604.12290-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2604.12290) [![Feishu](https://img.shields.io/badge/Feishu-Join-3370FF?style=flat-square)](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hxeVhZNN)

## News

- 🎉🚀 **2026-04-10**: **Frontier-Eng v1.0.0** (**47 tasks**) is out! [Detailed leaderboard](https://lab.einsia.ai/frontier-eng/leaderboard.html)

## Motivation

Existing agent benchmarks focus on binary pass/fail tasks (code generation, question answering). Engineering is different: the goal is rarely to produce a single correct artifact from scratch. An initial feasible solution already exists and the challenge is to **iteratively optimize it** under domain-specific constraints with no known theoretical ceiling.

Frontier-Eng formalizes this as **generative optimization** and identifies three limitations of prior benchmarks:

1. **Binary rewards**: 0/1 evaluation cannot measure incremental improvement or budget-aware search.
2. **Domain narrowness**: CS-only or abstracted-math tasks strip away real-world simulators, physical constraints, and domain knowledge that make engineering problems hard.
3. **Average-performance bias**: Engineering value is captured by **peak performance** — the best solution an agent can find within a fixed interaction budget — not aggregate accuracy across tasks.

Frontier-Eng evaluates agents on problems where genuine improvement requires integrating domain knowledge, constrained code synthesis, and iterative refinement against frozen, read-only verifiers.

## Getting Started

Setup is split between a small **driver** conda env and per-task **runtime** envs.

- **Driver** (`frontier-eval-2`): from `init.sh`; schedules jobs only.
- **Runtimes** (`frontier-v1-main`, `frontier-v1-kernel`, …): where benchmarks actually run. Install merged runtimes with `bash scripts/setup_v1_merged_task_envs.sh`.
- Before long runs: `export PYTHONNOUSERSITE=1` so user-site packages do not leak into tasks.
- Default task launch uses `task.runtime.use_conda_run=false` and `task.runtime.python_path=conda-env:<env_name>`.

**Task-specific bits**

- **DuckDB / EV2Gym**: need their local verifier deps (see each task dir).
- **Optics**: extra requirements under `benchmarks/Optics/` (also reflected in merged configs).
- **MolecularMechanics**: OpenFF stack (e.g. `openff-toolkit`); see task README.
- **GPU kernel tasks** (FlashAttention, MLA, …): need `frontier-v1-kernel`.

**External assets**

- **`dc-rl`**: clone + patch; paths under `third_party/` and `benchmarks/SustainableDataCenterControl/.../sustaindc/`.
- **PhySense**, **SustainDC**, **CarAerodynamicsSensing**: need downloaded models/data/checkpoints or they fail at runtime.

**Known issues**

- **ReactionOptimisation**: `frontier-v1-summit` pip resolution can fail; treat as env noise, not necessarily a bug in the harness.
- **EngDesign**: Docker tasks need a working Docker setup; use local mode if you cannot access the socket.

**LLM / API keys**: copy `.env.example` to `.env` and set at least **`OPENAI_API_KEY`** (and `OPENAI_API_BASE` / `OPENAI_MODEL` if you use a compatible gateway). Details: **[run.md](run.md)**.

Per-task commands, batch matrices, and overrides: **[frontier_eval/README.md](frontier_eval/README.md)**. **v1 batch** wrapper and host notes: **[run.md](run.md)** (`bash scripts/run_v1_batch.sh`).

## Leaderboard

Detailed leaderboard: **[lab.einsia.ai/frontier-eng/leaderboard.html](https://lab.einsia.ai/frontier-eng/leaderboard.html)**. **Frontier Models** — mean within-task rank (47 tasks), shown in ascending average-rank order.

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

## Task Details

The full task list by domain is in **[TASK_DETAILS.md](TASK_DETAILS.md)**.

## Best Found Solutions

The best solutions produced by our agent runs (across experiments, algorithms, models, and tasks) are archived in **[baseline_archive/README.md](baseline_archive/README.md)**. These serve as reference baselines for the community.

## Community

[Feishu](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) · [Discord](https://discord.gg/hxeVhZNN)

## Contributing

We welcome new engineering tasks via Pull Requests. Sample requirements, directory layout, test commands, EVOLVE-BLOCK rules, and the GitHub workflow are in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Citation

```
@article{chi2026frontier,
  title={Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization},
  author={Chi, Yizhe and Hong, Deyao and Jiang, Dapeng and Luo, Tianwei and Yang, Kaisen and Zhang, Boshi and Cao, Zhe and Fan, Xiaoyan and He, Bingxiang and Hao, Han and others},
  journal={arXiv preprint arXiv:2604.12290},
  year={2026}
}
```
