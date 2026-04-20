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

## Getting Started & Pre-flight Checklist

Executing a full sweep successfully requires understanding our split-environment architecture and properly initializing task-local dependencies. **Passing `init.sh` is not sufficient; a successful 0-iteration smoke test defines an active environment.**

### 1. Environment Architecture & Isolation
We decouple the **driver** and **task runtime** environments:
- **Driver Env** (`frontier-eval-2`): Created via `init.sh`, this *only* schedules and dispatches evaluations. It does not run benchmark code.
- **Runtime Envs** (`frontier-v1-main`, `frontier-v1-kernel`, etc.): These represent the true execution contexts.
  - To install merged task envs, use: `bash scripts/setup_v1_merged_task_envs.sh`. 
  - *Note on Isolation:* Always set `export PYTHONNOUSERSITE=1` before full runs to prevent local user packages from breaking task isolation.
  - *Python Path vs Conda Run:* By default, tasks use `task.runtime.use_conda_run=false` with a direct `task.runtime.python_path=conda-env:<env_name>` to cleanly launch isolated processes via conda prefixes.

### 2. Task-Local Extra Dependencies
Not every task runs purely off the main environment out of the box. Some require explicit local configuration:
- **DuckDB & EV2Gym**: These will **hard-crash** without installing their respective task-local verification dependencies. Check their task directories.
- **Optics Tasks**: Requires its dedicated requirement spec (`benchmarks/Optics/requirements.txt`, which is now merged into standard configs).
- **MolecularMechanics**: Requires OpenFF binary toolkits (e.g., `openff-toolkit`). See its README.
- **GPU Kernel Tasks** (e.g., FlashAttention, MLA): Explicitly require the `frontier-v1-kernel` suite to be active.

### 3. External Assets & Checklist
Sometimes failures are due to missing external assets rather than code issues:
- **`dc-rl`**: Requires additional `clone + patch` execution (should be stored in `third_party/` and `benchmarks/SustainableDataCenterControl/.../sustaindc/`).
- **`PhySense`**, **`SustainDC`**, and **`CarAerodynamicsSensing`**: Require downloading external models, data, or checkpoints. They will fail otherwise, which is a symptom of missing assets.

> 🤖 **Recommended: One-Click Setup via AI Agents** 
> If you are using Claude Code or other autonomous agent tools, you can simply run this prompt in the root directory to bootstrap the tedious setup process:
> `Please read the repository instructions, run init.sh to configure the main environment, and download the required third_party dependencies (Make sure to pin ShinkaEvolve to commit 642664d, and clone the correct dc-rl repo in the SustainDC task directory).`

### 4. Known Instabilities & Permissions
- **ReactionOptimisation**: Currently unstable (`frontier-v1-summit` pip resolution depth errors). Do not interpret its failure as a core framework issue.
- **EngDesign / Docker Tasks**: Docker-based workflows require explicit docker socket permissions. Depending on your machine, you must switch to local mode if permissions are denied.

Once these conditions are satisfied, per-task runs, batch matrices, and runtime overrides can be managed normally. See **[frontier_eval/README.md](frontier_eval/README.md)**. For **v1 batch** runs (`bash scripts/run_v1_batch.sh`) and host setup notes, see **[run.md](run.md)**.

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

## Join the Community

Welcome to our developer community! Whether you want to discuss new engineering problem concepts, find task collaborators, or encounter technical issues, reach us via [Feishu](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) or [Discord](https://discord.gg/hxeVhZNN).

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
