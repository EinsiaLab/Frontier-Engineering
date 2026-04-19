# Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks through Generative Optimization

[English](README.md) | 简体中文

[![主页](https://img.shields.io/badge/主页-lab.einsia.ai-0969DA?style=flat-square&logo=homepage&logoColor=white)](https://lab.einsia.ai/frontier-eng/) [![arXiv](https://img.shields.io/badge/arXiv-2604.12290-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2604.12290) [![飞书](https://img.shields.io/badge/飞书-加入讨论群-3370FF?style=flat-square)](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) [![Discord](https://img.shields.io/badge/Discord-加入-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/hxeVhZNN)

## News

- 🎉🚀 **2026-04-10**：**v1.0.0（47 tasks）** 正式发布！[详细榜单](https://lab.einsia.ai/frontier-eng/leaderboard.html)

## 动机

现有 Agent Benchmark 大多聚焦于 binary pass/fail 任务（代码生成、问答检索）。工程问题有本质区别：目标从不是"从零生成一个正确答案"，而是在领域约束下对已有可行解进行**迭代优化**，且往往没有理论最优上限。

Frontier-Eng 将这种范式形式化为 **generative optimization**，并指出现有 Benchmark 的三项局限：

1. **Binary reward**：0/1 评估无法衡量渐进式改进或 budget-aware 搜索能力。
2. **领域局限**：仅覆盖 CS 或高度抽象的数学问题，剥离了真实工程问题中的仿真器、物理约束与领域知识。
3. **平均性能偏差**：工程价值体现在 **peak performance**——Agent 在固定 budget 内能找到的最优解——而非跨任务的平均精度。

Frontier-Eng 要求 Agent 在只读、不可篡改的 verifier 下，将领域知识、受约束代码合成与迭代 refinement 紧密结合。

## Getting Started & 飞行前检查单 (Pre-flight Checklist)

想要成功运行完整的 benchmark 集合，必须要了解环境架构并正确配置局部的任务依赖。**跑完 `init.sh` 并不代表配置成功；能在 unified 任务上成功跑通一次 0-iteration (smoke test) 才是环境已最小可运行验证的标准。**

### 1. 环境架构分层与隔离
我们将**调度器所在环境**与**任务执行环境**剥离：
- **Driver 环境** (`frontier-eval-2`)：通过 `init.sh` 创建，只负责调度和派发任务，不涉及具体运行。
- **Runtime 环境** (`frontier-v1-main`, `frontier-v1-kernel` 等)：这些是实际运行代码的上下文。
  - 通过 `bash scripts/setup_v1_merged_task_envs.sh` 来安装打包好的执行环境。
  - **隔离注意**：在进行全量测试前，请务必设置 `export PYTHONNOUSERSITE=1` 来防止本地用户包穿透。
  - **执行模式**：默认情况下，任务配置会使用 `task.runtime.use_conda_run=false` 并指定 `task.runtime.python_path=conda-env:<env_name>`，直接借助 conda prefix 来干净地启动进程。

### 2. Task-Local 额外依赖
有些 benchmark 不能光靠主环境直接跑通，还需对任务进行局部配置：
- **DuckDB 与 EV2Gym**：不安装其特定的验证依赖会直接挂掉，属于 "少装一步就会报错" 的典型。
- **Optics 任务族**：需要专属的依赖集（目前已并入主环境配置中）。
- **MolecularMechanics**：需依赖 OpenFF 系列的底层包（如 `openff-toolkit`），安装方法见其子目录 README。
- **GPU Kernel 类任务** (FlashAttention 等)：有单独的 GPU kernel 运行环境 (`frontier-v1-kernel`)，绝不能顺延使用主环境。

### 3. 外部资产与数据 Checklist
有些任务失败不是代码/依赖问题，而是缺少了对应的外部资产：
- **`dc-rl`** 需要额外执行 `clone + patch`（存放于 `third_party/` 与 `benchmarks/SustainableDataCenterControl/.../sustaindc/`）。
- **`PhySense`、`SustainDC` 及 `CarAerodynamicsSensing`**：需手动把运行所需的数据、外部模型权重或参考文件补齐，否则将被系统误判为失效。

> 🤖 **推荐使用 AI 助手一键完成繁琐的环境配置**：
> 如果你使用 Claude Code 或其他智能 Agent 工具，可以直接在项目根目录输入这段 Prompt：
> `请阅读 repo 说明，帮我执行 init.sh 配置主环境，并下载所需的 third_party 依赖（注意将 ShinkaEvolve 锁定在 642664d 版本，并在 SustainDC 目录 clone 正确的 dc-rl 库）。`

### 4. 权限限制与已知阻塞项
- **ReactionOptimisation**：当前环境依赖极其不稳定 (`frontier-v1-summit` 经常出现 pip resolution over depth 问题)，请把它当成已知阻塞项，并非你本地操作失误。
- **EngDesign / Docker 权限**：有些任务极度依赖 Docker 套接字权限，如果当前机器/节点没有 Docker 权限，你需要切换到替代的本地运行模式（Local Mode）。

完成上述准备后，即可查看详细说明进行单任务运行与大规模测试：**[frontier_eval/README_zh-CN.md](frontier_eval/README_zh-CN.md)**。需要 **v1 批量一键命令** 与主机侧注意事项时，见 **[run.md](run.md)**。

## Leaderboard

详细榜单：**[lab.einsia.ai/frontier-eng/leaderboard.html](https://lab.einsia.ai/frontier-eng/leaderboard.html)**。**Frontier Models** — 平均任务内排名（47 tasks，下表按平均排名从低到高展示）。

| 排名 | Model | Average Rank |
| :--: | :--- | --: |
| 1 | Claude Opus 4.6 | 3.18 |
| 2 | GLM-5 | 4.02 |
| 3 | DeepSeek V3.2 | 4.41 |
| 4 | GPT-OSS-120B | 4.46 |
| 5 | Gemini 3.1 Pro Preview | 5.34 |
| 6 | Grok 4.20 | 5.60 |
| 7 | SEED 2.0 Pro | 5.63 |
| 8 | GPT-5.4 | 5.68 |
| 9 | Qwen3 Coder Next | 6.68 |

## 任务详情

完整任务列表见 **[TASK_DETAILS_zh-CN.md](TASK_DETAILS_zh-CN.md)**。

## 最优解存档

我们在各实验 / 算法 / 模型 / task 组合上跑出的最优代码存档于 **[baseline_archive/README.md](baseline_archive/README.md)**，可作为社区参考 baseline。

## 加入社区

欢迎加入我们的开发者社区！无论是讨论新的工程问题构想、寻找 task 合作者，还是遇到技术问题，可通过[飞书](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c)或[Discord](https://discord.gg/hxeVhZNN)直接联系我们。

## 贡献指南

欢迎通过 Pull Request 提交新的工程任务。样本要求、目录结构、测试命令、EVOLVE-BLOCK 规则与 GitHub 协作流程见 **[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md)**。
