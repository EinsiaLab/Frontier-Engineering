# Frontier-Eng

[English](README.md) | 简体中文

[主页](https://lab.einsia.ai/frontier-eng/) · [arXiv](http://arxiv.org/abs/2604.12290) · [Discord](https://discord.gg/hxeVhZNN)

Frontier-Eng 是一个面向 **generative optimization** 的 benchmark：Agent 不是一次性写出“标准答案”，而是持续修改可运行的工程代码，读取只读 verifier 的反馈，并在固定预算内不断改进。

当前版本包含 **47 个任务**，覆盖计算系统、量子信息、运筹优化、机器人控制、光学通信、物理与工程设计。主页和论文的核心观点是：真实工程问题通常从一个可行 baseline 出发，价值来自持续优化，而不是 pass/fail。

## 这个 benchmark 在测什么

和传统 agent benchmark 相比，Frontier-Eng 更关注三件事：

- 连续分数，而不是二值对错
- 真正的 verifier / simulator，而不是 judge model
- 在预算内能把 baseline 推到多远，而不是平均通过率


## 0. Host Requirements

仓库自带的 Python 环境可以自动化，但如果你要完整跑 `v1` baseline sweep，宿主机仍然默认需要具备下面这些条件：

| 条件 | 用途 | 说明 |
|---|---|---|
| 常规构建工具和可用外网 | 所有 setup 路径 | `uv`、Python wheel 和 benchmark-local 下载都会用到 |
| NVIDIA GPU + 可用 CUDA 栈 | `KernelEngineering/*`、`Aerodynamics/CarAerodynamicsSensing`、部分 Robotics 任务 | 纯 CPU 任务不需要 |
| Docker | `EngDesign` | 只有这一组任务依赖 |
| Octave | `Astrodynamics/MannedLunarLanding` | 属于 `uv` 不会安装的宿主机工具 |
| 额外数据 / checkpoint / 第三方仓库 | 如 `SustainableDataCenterControl`、`CarAerodynamicsSensing`、`MolecularMechanics` | 具体路径和准备方式见各任务 README |

如果你想看更具体的宿主机准备步骤，包括 `Docker` / `Octave` 怎么装、哪些外部资产要单独准备，请先读 [`run_zh-CN.md`](run_zh-CN.md)。

## 快速开始

### 1. 安装 `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. 创建 driver 环境

在仓库根目录执行：

```bash
bash init.sh
source .venvs/frontier-eval-driver/bin/activate
```

这里创建的是 **driver** 环境，只负责运行 `python -m frontier_eval`。

### 3. 先跑 smoke

```bash
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

这个命令不需要 LLM key，适合先确认框架本身没问题。

## 跑真实 benchmark

这个仓库的环境分两层：

- `frontier-eval-driver`：driver 环境
- `.venvs/<runtime-name>`：任务 runtime，比如 `frontier-v1-main`、`frontier-v1-kernel`、`frontier-v1-summit`

创建发布版 `v1` 题集使用的 task runtime：

```bash
bash scripts/env/setup_v1_task_envs.sh
```

现在 runtime 选择方式是：

- `task.runtime.env_name=<name>`：把 `.venvs/<name>/bin` 放到 `PATH` 前面
- `task.runtime.python_path=uv-env:<name>`：需要显式解释器路径的任务，直接解析到 `.venvs/<name>/bin/python`

### 单任务 baseline 验证

不需要 LLM key：

```bash
python -m frontier_eval \
  task=unified \
  task.benchmark=WirelessChannelSimulation/HighReliableSimulation \
  algorithm=openevolve \
  algorithm.iterations=0
```

### 整套 `v1` baseline 验证

如果你想验证发布版 `v1` 题集的 baseline，而不是跑优化过程：

```bash
bash scripts/batch/validate_v1_task_envs.sh
```

它会按 `v1` 题集对应的 batch 配置把任务都以 `algorithm.iterations=0` 跑一遍，也就是只评估仓库自带 baseline，不会调用 LLM。

如果你后面要跑正常的优化实验，再看 [`run_zh-CN.md`](run_zh-CN.md)。

## 继续阅读

- 框架命令与 task onboarding：[`frontier_eval/README_zh-CN.md`](frontier_eval/README_zh-CN.md)
- `v1` 题集的批量运行说明：[`run_zh-CN.md`](run_zh-CN.md)
- 完整任务列表：[`TASK_DETAILS_zh-CN.md`](TASK_DETAILS_zh-CN.md)
- 历史实验最优代码存档：[`baseline_archive/README.md`](baseline_archive/README.md)

## Leaderboard

详细榜单见 [lab.einsia.ai/frontier-eng/leaderboard.html](https://lab.einsia.ai/frontier-eng/leaderboard.html)。

| 排名 | Model | Average Rank |
| :--: | :--- | --: |
| 1 | Claude Opus 4.6 | 3.18 |
| 2 | GLM-5 | 4.02 |
| 3 | DeepSeek V3.2 | 4.41 |
| 4 | Gemini 3.1 Pro Preview | 5.34 |
| 5 | Grok 4.20 | 5.60 |
| 6 | SEED 2.0 Pro | 5.63 |
| 7 | GPT-5.4 | 5.68 |
| 8 | Qwen3 Coder Next | 6.68 |

## 贡献

贡献指南见 [`CONTRIBUTING_zh-CN.md`](CONTRIBUTING_zh-CN.md)。
