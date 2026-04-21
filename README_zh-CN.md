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

主页和论文里最值得记住的结论有三个：

1. 改进出现的频率大致按 `1 / iteration` 衰减。
2. 每一次改进的幅度大致按 `1 / improvement count` 衰减。
3. 在总预算固定时，一条更深的搜索链通常优于很多条浅层重启。

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
source .venvs/frontier-eval-2/bin/activate
```

这里创建的是 **driver** 环境，只负责运行 `python -m frontier_eval`。

### 3. 先跑 smoke

```bash
python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0
```

这个命令不需要 LLM key，适合先确认框架本身没问题。

## 跑真实 benchmark

这个仓库的环境分两层：

- `frontier-eval-2`：driver 环境
- `.venvs/<runtime-name>`：任务 runtime，比如 `frontier-v1-main`、`frontier-v1-kernel`、`frontier-v1-summit`

创建 `v1` 发布矩阵使用的合并 runtime：

```bash
bash scripts/setup_v1_merged_task_envs.sh
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

如果你想验证发布版 `v1` 的 baseline，而不是跑优化过程：

```bash
bash scripts/validate_v1_merged_task_envs.sh
```

它会把 batch matrix 里的任务都以 `algorithm.iterations=0` 跑一遍，也就是只评估仓库自带 baseline，不会调用 LLM。

如果你后面要跑正常的优化实验，再看 [`run_zh-CN.md`](run_zh-CN.md)。

## 哪些东西仍然要手动准备

`uv` 现在负责仓库内的 Python 环境，但有些任务仍然依赖 Python 之外的系统组件或外部资源：

- **Octave 类任务**：主机上要单独安装 `octave`
- **GPU kernel 类任务**：要有可用的 CUDA 环境
- **EngDesign**：通常需要 Docker 和额外的容器配置
- **SustainableDataCenterControl**：需要 vendored `dc-rl` 和任务资产
- **CarAerodynamicsSensing**、**PhySense**、**SustainDC**：需要单独下载模型或数据
- **MolecularMechanics**：截至 2026 年，OpenFF 工具链还不能只靠 `uv` 完整复现，所以这组 runtime 仍然需要手动准备

这些 benchmark-local 的前置条件，请以 `benchmarks/` 目录下各任务 README 为准。

## 继续阅读

- 框架命令与 task onboarding：[`frontier_eval/README_zh-CN.md`](frontier_eval/README_zh-CN.md)
- `v1` 批量运行说明：[`run_zh-CN.md`](run_zh-CN.md)
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
