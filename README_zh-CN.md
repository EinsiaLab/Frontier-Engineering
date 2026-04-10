# Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks through Generative Optimization

[English](README.md) | 简体中文

[![主页](https://img.shields.io/badge/主页-lab.einsia.ai-0969DA?style=flat-square&logo=homepage&logoColor=white)](https://lab.einsia.ai/frontier-eng/) [![arXiv](https://img.shields.io/badge/arXiv-即将发布-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org) [![飞书](https://img.shields.io/badge/飞书-加入讨论群-3370FF?style=flat-square)](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) [![Discord](https://img.shields.io/badge/Discord-加入-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hxeVhZNN)

## 动机

现有 Agent Benchmark 大多聚焦于 binary pass/fail 任务（代码生成、问答检索）。工程问题有本质区别：目标从不是"从零生成一个正确答案"，而是在领域约束下对已有可行解进行**迭代优化**，且往往没有理论最优上限。

Frontier-Eng 将这种范式形式化为 **generative optimization**，并指出现有 Benchmark 的三项局限：

1. **Binary reward**：0/1 评估无法衡量渐进式改进或 budget-aware 搜索能力。
2. **领域局限**：仅覆盖 CS 或高度抽象的数学问题，剥离了真实工程问题中的仿真器、物理约束与领域知识。
3. **平均性能偏差**：工程价值体现在 **peak performance**——Agent 在固定 budget 内能找到的最优解——而非跨任务的平均精度。

Frontier-Eng 要求 Agent 在只读、不可篡改的 verifier 下，将领域知识、受约束代码合成与迭代 refinement 紧密结合。

## Getting Started

需要已安装 **[conda](https://docs.conda.io/en/latest/miniconda.html)**，并在 **Bash** 里执行（Windows 可用 Git Bash、WSL；一般不要在未配置 `bash` 的纯 `cmd` / PowerShell 里直接跑）。

```bash
bash init.sh && conda activate frontier-eval-2
```

运行具体任务、batch、环境覆盖等见 **[frontier_eval/README_zh-CN.md](frontier_eval/README_zh-CN.md)**。

## Leaderboard

交互榜单：**[lab.einsia.ai/frontier-eng/leaderboard.html](https://lab.einsia.ai/frontier-eng/leaderboard.html)**。**Frontier Models** — 归一化总分（47 tasks）。*2026-04-09 · [数据](https://lab.einsia.ai/frontier-eng/data/overall-model.yaml)*

| 排名 | Model | Score |
| :--: | :--- | --: |
| 1 | Claude Opus 4.6 | 0.751 |
| 2 | GLM-5 | 0.630 |
| 3 | DeepSeek V3.2 | 0.601 |
| 4 | Gemini 3.1 Pro Preview | 0.442 |
| 5 | SEED 2.0 Pro | 0.437 |
| 6 | Grok 4.20 | 0.436 |
| 7 | GPT-5.4 | 0.433 |
| 8 | Qwen3 Coder Next | 0.272 |

## Agent skill

请把下面这段话**整段复制**，发给你的 Agent：

```text
请你把本仓库 skill/source/ 下的两个 skill 安装为**本仓库的项目级 skill**（按我当前客户端对项目级 skill 的目录约定处理，例如 Cursor 的 .cursor/skills/、Claude Code 的 .claude/skills/、Codex 的 .codex/skills/ 等），分别是：

1. skill/source/frontier-evaluator — 协助运行与调试 frontier_eval 评测、按各 benchmark README 准备运行环境与命令；
2. skill/source/frontier-contributor — 协助按本仓库规范贡献或更新 benchmark。

每个目录请以「一个可加载的 skill 包」接入：至少包含该目录下的 SKILL.md，以及其中引用的相对路径资源（例如 frontier-evaluator 下的 scripts/ 等），保证安装后在本仓库根目录下可正常使用。
```

仓库不提供统一的 skill CLI；以各目录内 `SKILL.md` 为准，由你的 Agent 按客户端接入。

## News

- 2026-04-10: 发布 Frontier-Eng 1.0.0! 榜单与图表见 [leaderboard.html](https://lab.einsia.ai/frontier-eng/leaderboard.html) 与 [lab.einsia.ai/frontier-eng/](https://lab.einsia.ai/frontier-eng/)。

## 任务详情

完整任务列表见 **[TASK_DETAILS_zh-CN.md](TASK_DETAILS_zh-CN.md)**。

## 最优解存档

我们在各实验 / 算法 / 模型 / task 组合上跑出的最优代码存档于 **[baseline_archive/README.md](baseline_archive/README.md)**，可作为社区参考 baseline。

## 加入社区

欢迎加入我们的开发者社区！无论是讨论新的工程问题构想、寻找 task 合作者，还是遇到技术问题，可通过页面顶部的链接联系我们。

## 贡献指南

欢迎通过 Pull Request 提交新的工程任务。样本要求、目录结构、测试命令、EVOLVE-BLOCK 规则与 GitHub 协作流程见 **[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md)**。
