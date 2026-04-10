# Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks through Generative Optimization

English | [简体中文](README_zh-CN.md)

[![Homepage](https://img.shields.io/badge/Homepage-lab.einsia.ai-0969DA?style=flat-square&logo=homepage&logoColor=white)](https://lab.einsia.ai/frontier-eng/) [![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org) [![Feishu](https://img.shields.io/badge/Feishu-Join-3370FF?style=flat-square)](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) [![Discord](https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hxeVhZNN)

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

```bash
bash init.sh && conda activate frontier-eval-2
```

Per-task runs, batch matrices, and runtime overrides are in **[frontier_eval/README.md](frontier_eval/README.md)**.

## Leaderboard

Detailed leaderboard: **[lab.einsia.ai/frontier-eng/leaderboard.html](https://lab.einsia.ai/frontier-eng/leaderboard.html)**. **Frontier Models** — normalized overall score (47 tasks). *2026-04-09 · [source](https://lab.einsia.ai/frontier-eng/data/overall-model.yaml)*

| Rank | Model | Score |
| :--: | :--- | --: |
| 1 | Claude Opus 4.6 | 0.751 |
| 2 | GLM-5 | 0.630 |
| 3 | DeepSeek V3.2 | 0.601 |
| 4 | Gemini 3.1 Pro Preview | 0.442 |
| 5 | SEED 2.0 Pro | 0.437 |
| 6 | Grok 4.20 | 0.436 |
| 7 | GPT-5.4 | 0.433 |
| 8 | Qwen3 Coder Next | 0.272 |

## Agent skills

**Copy the block below verbatim** and send it to your agent:

```text
Please install the two skills under this repo’s skill/source/ as project-level skills for this repository (follow my editor’s / client’s convention for where project skills live, e.g. .cursor/skills/ for Cursor, .claude/skills/ for Claude Code, .codex/skills/ for Codex, etc.). The two packages are:

1. skill/source/frontier-evaluator — help run and debug frontier_eval evaluations and prepare per-benchmark runtime from each benchmark’s README;
2. skill/source/frontier-contributor — help add or update benchmarks following this repo’s contribution rules.

Install each folder as one loadable skill package: include SKILL.md plus any relative assets it references (e.g. scripts/ under frontier-evaluator) so everything works from the repository root after installation.
```

There is no bundled CLI installer; your agent wires these up per client. The authoritative instructions remain in each folder’s `SKILL.md`.

## Task Details

The full task list by domain is in **[TASK_DETAILS.md](TASK_DETAILS.md)**.

## Best Found Solutions

The best solutions produced by our agent runs (across experiments, algorithms, models, and tasks) are archived in **[baseline_archive/README.md](baseline_archive/README.md)**. These serve as reference baselines for the community.

## Join the Community

Welcome to our developer community! Whether you want to discuss new engineering problem concepts, find task collaborators, or encounter technical issues, reach us via the links at the top of this page.

## Contributing

We welcome new engineering tasks via Pull Requests. Sample requirements, directory layout, test commands, EVOLVE-BLOCK rules, and the GitHub workflow are in **[CONTRIBUTING.md](CONTRIBUTING.md)**.
