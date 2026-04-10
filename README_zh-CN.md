<div align="center">

# Frontier-Engineering Bench

**用真实工程优化问题评测前沿 AI——不是玩具题。**

[![任务数](https://img.shields.io/badge/任务-95+-blue?style=flat-square)](benchmarks/)
[![领域数](https://img.shields.io/badge/领域-24个-purple?style=flat-square)](benchmarks/)
[![榜单](https://img.shields.io/badge/榜单-在线-brightgreen?style=flat-square)](https://einsia.github.io/Einsia-lab/frontier-eng/)
[![欢迎PR](https://img.shields.io/badge/PR-欢迎贡献-orange?style=flat-square)](CONTRIBUTING.md)

[English](README.md) | 简体中文

</div>

---

## 这是什么？

不同于把工程问题简化为数学题的 Benchmark，Frontier-Eng 让 AI Agent 面对具有**真实物理约束和经济价值**的问题——结构承载、化学产率、轨道力学、GPU 核心吞吐量。要么找到更优解，要么找不到。

三点核心差异：

- **开放式评分** — 持续改进比一次性通过更重要
- **极值导向** — 关注模型能达到的最优解，而非平均表现
- **真实复杂度** — 保留领域知识、约束条件和工具调用

## 任务覆盖

| 领域 | 任务数 | 领域 | 任务数 |
|------|------:|------|------:|
| 🔬 光学 | 17 | 🏭 车间调度 | 10 |
| 🏗️ 工程设计 | 9 | 🤖 机器人 | 6 |
| ⚗️ 反应优化 | 5 | 📦 库存优化 | 5 |
| 🧬 分子力学 | 4 | 🏛️ 结构优化 | 4 |
| 🔬 单细胞分析 | 4 | 📡 通信工程 | 3 |
| 🔐 密码学 | 3 | 🔌 电子设计自动化 | 3 |
| ⚙️ 内核工程 | 3 | 📈 投资组合优化 | 3 |
| ⚛️ 量子计算 | 3 | ✈️ 空气动力学 | 2 |
| 💻 计算机系统 | 2 | ⚡ 储能 | 2 |
| 🌍 粒子物理 | 2 | 🚀 轨道动力学 | 1 |
| 🔩 增材制造 | 1 | 🔋 电力系统 | 1 |
| 🌿 绿色数据中心 | 1 | 📶 无线信道仿真 | 1 |

→ 完整任务列表与进度：[TASK_PROGRESS_zh-CN.md](TASK_PROGRESS_zh-CN.md)

## 快速开始

```bash
# 运行评测框架
conda activate frontier-eval-2
python -m frontier_eval task=unified task.benchmark=Robotics/PIDTuning algorithm.iterations=5

# 安装 Agent skill
python -m frontier_eval skill evaluator codex
```

<details>
<summary>⚙️ 运行环境说明</summary>

`frontier_eval/requirements.txt` 只负责搭建评测框架本身——并非所有 benchmark 都能在同一环境中运行。

**运行任何 benchmark 前，请先阅读对应的 README**（`benchmarks/<Domain>/README.md` 或 `benchmarks/<Domain>/<Task>/README.md`）。README 中若有 `task.runtime.conda_env=...`、`task.runtime.use_conda_run=false` 等 override，以 README 为准。

**v1 合并环境：**

| 环境 | 包含任务 |
|---|---|
| `frontier-v1-main` | QuantumComputing、Optics、InventoryOptimization、PyPortfolioOpt、JobShop、大部分 Robotics、Aerodynamics、KernelEngineering/FlashAttention |
| `frontier-v1-summit` | ReactionOptimisation/\* |
| `frontier-v1-sustaindc` | SustainableDataCenterControl/\* |
| `frontier-v1-kernel` | KernelEngineering/MLA、KernelEngineering/TriMul |

环境初始化：`bash scripts/setup_v1_merged_task_envs.sh`  
验证：`DRIVER_ENV=frontier-eval-2 GPU_DEVICES=<id> bash scripts/validate_v1_merged_task_envs.sh`

</details>

## 贡献新任务

欢迎通过 PR 提交新的工程问题。三个核心要求：

1. **贴近现实** — 必须包含真实的物理/工程约束，不能是纯数学题
2. **有经济价值** — 解决它在实践中要有意义
3. **可验证** — 提供可运行的 `evaluator.py`（优先 Docker）

> 欢迎 AI 辅助贡献。运行 `python -m frontier_eval skill` 安装 `Contributor` skill，但请保持人工审查介入，不要完全交给 AI。

<details>
<summary>📁 任务文件结构</summary>

```
<Domain>/
├── README.md                    # 领域综述
└── <TaskName>/
    ├── README.md                # 运行说明、快速开始
    ├── Task.md                  # 背景、模型、输入输出定义
    ├── references/              # constants.json、文献等
    ├── frontier_eval/           # unified task 元数据
    │   ├── initial_program.txt
    │   ├── eval_command.txt
    │   ├── agent_files.txt
    │   └── artifact_files.txt
    ├── verification/
    │   ├── evaluator.py         # 评分入口
    │   ├── requirements.txt
    │   └── docker/Dockerfile
    └── baseline/                # 可选：参考解法
```

新任务必须使用 unified 格式（`task=unified`）。完整元数据说明见 `frontier_eval/README_zh-CN.md`。

**EVOLVE-BLOCK 标记**（ShinkaEvolve / ABMCTS 必须）——在被 Agent 演化的文件中标记可编辑区域：
```python
# EVOLVE-BLOCK-START
...你的代码...
# EVOLVE-BLOCK-END
```

</details>

<details>
<summary>🔀 贡献流程</summary>

1. Fork → 创建分支 `feat/<Domain>/<TaskName>`
2. 按上述结构添加任务文件
3. 本地测试：`python verification/evaluator.py scripts/init.py`
4. 框架适配验证：`python -m frontier_eval task=unified task.benchmark=<Domain>/<Task> algorithm.iterations=0`
5. 提交 PR，简要说明任务背景和运行方式
6. AI Agent 自动初审，维护者最终合并

> 第一次贡献？先开 Issue 讨论你的任务方向。

</details>

<details>
<summary>📦 best_code_only 快照</summary>

`best_code_only/` 收录各实验 / 算法 / 模型 / task 的最终全局最优代码。

```
best_code_only/
└── <experiment>/<algorithm>/<model>/<task>/<code-file>
```

先查 `best_code_only/coverage.json` 确认覆盖范围。示例路径：
- `best_code_only/experiment1/openevolve/gpt-5.4/Astrodynamics_MannedLunarLanding/`
- `best_code_only/experiment2/shinkaevolve/claude-opus-4.6/KernelEngineering_TriMul/`

</details>

## 评测框架

核心实现：[`frontier_eval/`](frontier_eval/)——完整使用说明见[评测 README](frontier_eval/README_zh-CN.md)。部分算法需要在 `third_party/` 下 clone 外部仓库。

## 加入社区

| | |
|---|---|
| 🟢 飞书 | [加入讨论群](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c) |
| 💬 Discord | [加入社区](https://discord.gg/hxeVhZNN) |

<!-- AI_GENERATED -->
