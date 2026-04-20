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

## 上手

环境分两层：**调度用的 driver conda** 和 **各任务 runtime**。

- **Driver**（`frontier-eval-2`）：`init.sh` 创建，只负责调度。
- **Runtime**（`frontier-v1-main`、`frontier-v1-kernel` 等）：任务真正跑在这里。合并安装：`bash scripts/setup_v1_merged_task_envs.sh`。
- 长时间跑之前建议：`export PYTHONNOUSERSITE=1`，避免本机用户目录里的包混进任务进程。
- 默认用 `task.runtime.use_conda_run=false` 和 `task.runtime.python_path=conda-env:<env_name>` 启进程。

**按任务额外准备**

- **DuckDB / EV2Gym**：要装各自任务目录里写的校验依赖。
- **Optics**：见 `benchmarks/Optics/` 依赖说明（合并配置里也会带上）。
- **MolecularMechanics**：OpenFF 等，见该任务 README。
- **GPU kernel 类**（FlashAttention 等）：需要 `frontier-v1-kernel`，不能只用主 env。

**外部资源**

- **`dc-rl`**：按说明 clone + patch，路径在 `third_party/` 与 `benchmarks/SustainableDataCenterControl/.../sustaindc/`。
- **PhySense、SustainDC、CarAerodynamicsSensing**：要自备数据、模型或权重，缺了会跑失败。

**已知问题**

- **ReactionOptimisation**：`frontier-v1-summit` 上 pip 解析可能炸，优先当环境/依赖问题看。
- **EngDesign**：依赖 Docker；没权限就改用文档里的本地模式。

单任务、批量矩阵与覆盖项：**[frontier_eval/README_zh-CN.md](frontier_eval/README_zh-CN.md)**。v1 批量脚本与主机说明：**[run.md](run.md)**（`bash scripts/run_v1_batch.sh`）。

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

## 社区

[飞书](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) · [Discord](https://discord.gg/hxeVhZNN)

## 贡献指南

欢迎通过 Pull Request 提交新的工程任务。样本要求、目录结构、测试命令、EVOLVE-BLOCK 规则与 GitHub 协作流程见 **[CONTRIBUTING_zh-CN.md](CONTRIBUTING_zh-CN.md)**。
