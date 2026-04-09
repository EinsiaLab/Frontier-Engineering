# Frontier-Eng: 人工智能代理的大规模工程优化基准

English | [简体中文](README_zh-CN.md)

**Frontier-Eng** 是一个旨在评估 AI Agent 在**真实工程领域**中解决**开放式优化问题**能力的Benchmark。

不同于现有的关注计算机科学（CS）或纯数学抽象问题的 Benchmark，Frontier-Eng 聚焦于具有实际**经济效益**和**物理约束**的工程难题，预期涵盖航天、土木、EDA、生物工程等多个领域。

## 运行环境说明

`frontier_eval/requirements.txt` 只负责把评测框架本身装起来，并不代表所有 benchmark 都能在同一个环境里直接运行。

在运行任何具体 benchmark 之前，请先阅读对应的环境说明：

- `benchmarks/<Domain>/README*.md`
- 如果该 task 还有自己的 README，再继续看 `benchmarks/<Domain>/<Task>/README*.md`

很多 benchmark 家族都需要自己的 runtime 环境、额外的 `requirements.txt`、额外的 `third_party/` checkout，或者 Docker 执行方式。只要 benchmark README 里写了 `task.runtime.conda_env=...`、`task.runtime.python_path=...`、`task.runtime.use_conda_run=false` 这类 runtime override，就应当以 benchmark README 为准，并把这些 override 原样带进运行命令。

仓库里已经有多类不同模式，例如 `ReactionOptimisation`（`summit`）、`MolecularMechanics`（`openff-dev`）、`SustainableDataCenterControl`（`sustaindc`）、`PyPortfolioOpt`（`pyportfolioopt`）、`QuantumComputing`（`mqt`）、`InventoryOptimization`（`stock`）、`JobShop`（自定义 `python_path`）以及 `EngDesign`（Docker / 本地模式）。

仓库内已经打包了项目级 agent skill，位于 `skill/`。运行 `python -m frontier_eval skill` 可交互式选择；也可以用 `python -m frontier_eval skill evaluator codex` 直接安装。

### `v1` 合并任务环境说明

为减少 `README` 中标记为有效 `v1` 任务池的 task runtime 环境数量，同时不破坏已有环境，仓库当前采用如下约定：

- `frontier-eval-2` 仍然只作为评测框架 / driver 环境使用，保持不变。
- 原有 task 环境（如 `bio`、`mqt`、`optics`、`stock`、`pyportfolioopt`、`motion`、`jobshop`、`summit`、`sustaindc`、`kernel`）不会被删除或覆盖。
- 新增的合并环境会创建在当前 `conda` 指向的环境目录下，默认环境名为 `frontier-v1-main`、`frontier-v1-summit`、`frontier-v1-sustaindc`、`frontier-v1-kernel`。
- 对于需要“直连解释器”而不是 `conda run` 的 `v1` 任务（当前主要是 `ReactionOptimisation/*` 与 `JobShop/*`），batch matrix 里使用可移植标记 `conda-env:<env-name>`，由 unified evaluator 在运行时解析为对应环境中的 Python 路径，因此不需要把机器本地前缀写进仓库。

当前 `v1` task runtime 合并结果为：

- `frontier-v1-main`：`SingleCellAnalysis/predict_modality`、`QuantumComputing/*`、`Optics/*`、`InventoryOptimization/*`、`PyPortfolioOpt/*`、`JobShop/*`、`Robotics/DynamicObstacleAvoidanceNavigation`、`Robotics/PIDTuning`、`Robotics/UAVInspectionCoverageWithWind`、`Robotics/QuadrupedGaitOptimization`、`Robotics/RobotArmCycleTimeOptimization`、`Aerodynamics/CarAerodynamicsSensing`、`KernelEngineering/FlashAttention`
- `frontier-v1-summit`：`ReactionOptimisation/*`
- `frontier-v1-sustaindc`：`SustainableDataCenterControl/*`
- `frontier-v1-kernel`：`KernelEngineering/MLA`、`KernelEngineering/TriMul`

如果某个历史 README 仍然写着旧环境名（例如 `mqt`、`stock`、`pyportfolioopt`、`jobshop` 等），对于当前 `v1` 批量运行，请优先以 `frontier_eval/conf/batch/` 下的 matrix 配置为准。

环境准备与验证脚本：

- 初始化合并环境：`bash scripts/setup_v1_merged_task_envs.sh`
- 按 `iter=0` 验证合并环境：`DRIVER_ENV=frontier-eval-2 GPU_DEVICES=<gpu_id> bash scripts/validate_v1_merged_task_envs.sh`

说明：

- 上述验证默认使用 `conda run -n frontier-eval-2 python` 作为 driver，也可以通过 `DRIVER_PY=/path/to/python` 显式覆盖；脚本会验证 CPU `v1`、GPU `v1`、`FlashAttention`、`MLA`、`TriMul`。
- `MuonTomography` 仍按本文后续说明，暂不计入当前有效 `v1` 任务池。
- 已知限制：`KernelEngineering/TriMul` 的官方 full benchmark (`verification/tri_bench.txt`) 在 24GB 级别 GPU 上可能受显存上限影响；这通常是 task 本身的显存边界问题，而不是 `frontier-v1-kernel` 环境缺依赖。

## 🎯 动机

当前的 AI4Research 评测体系存在以下局限性：

1. **评估方式单一**：大多采用 0/1 二元评估或封闭区间的 Rubric，无法有效衡量 Agent 在开放世界中通过交互进行**迭代优化**的能力。
2. **领域局限**：现有 Benchmark 大多局限于 CS 领域（如代码生成），或将实际问题高度抽象为数学题，剥离了现实世界的复杂性，使得 Agent 无法利用丰富的外部知识和工具。
3. **指标偏差**：传统计算指标关注模型的平均表现，而对于工程优化问题，我们更应关注模型在单一问题上通过探索机制所能达到的**极值（Peak Performance）**。

**Frontier-Eng** 旨在通过提供丰富的上下文和工具支持，评估 Agent 在广泛工程学科中解决具有实际价值问题的能力。

## 🤝 贡献指南

我们需要社区的力量来扩展 Benchmark 的覆盖范围。我们欢迎通过 Pull Request (PR) 的方式提交新的工程问题。如果你希望贡献，请遵循以下标准和流程：

> **AI 辅助贡献**：我们欢迎使用 AI 工具辅助创建的贡献。如果你使用 agent 来协助贡献，建议运行 `python -m frontier_eval skill` 安装 `Contributor`，或者直接把 `skill/source/frontier-contributor/SKILL.md` 作为提示词来源。**但是，请不要过度依赖 AI 工具或完全放手不管**。人工审查和监督对于确保质量和正确性至关重要。

### 样本要求

1. **Reality Gap**: 必须贴近现实，考虑现实影响因素，非单纯数学抽象。
2. **Economic Value**: 问题解决后应具有明确的工程或经济价值。
3. **Verifiability**: 必须提供可执行的验证程序（Docker 优先），能在可接受时间内完成评测。

### 提交格式

每一个 Task 应当包含以下文件结构：

```text
<Domain_Name>/                       # 一级目录：领域名称 (e.g., Astrodynamics)
├── README.md                        # [必选] 领域综述 (默认入口，中英文均可)：介绍背景及子任务索引
├── README_zh-CN.md                  # [可选] 领域综述 (中文版。仅当 README.md 为英文且提供了中文版时使用)
├── <Task_Name_A>/                   # 二级目录：具体任务名称 (e.g., MannedLunarLanding)
│   ├── README.md                    # [必选] 导航文档：说明文件结构、如何运行及快速开始
│   ├── README_zh-CN.md              # [可选] 导航文档
│   ├── Task.md                      # [必选] 任务详情文档：核心文档，包含背景、物理模型、输入输出定义
│   ├── Task_zh-CN.md                # [可选] 任务详情文档
│   ├── references/                  # 参考资料目录
│   │   ├── constants.json           # 物理常数、仿真参数等
│   │   └── manuals.pdf              # 领域知识手册、物理方程或约束条件文档
│   ├── frontier_eval/               # [必选] Frontier Eval 的 unified-task 元数据
│   │   ├── initial_program.txt      # 初始可编辑程序路径（相对任务根目录）
│   │   ├── eval_command.txt         # `task=unified` 使用的评测命令模板
│   │   ├── agent_files.txt          # 提供给 Agent 的上下文文件
│   │   ├── artifact_files.txt       # 评测后需要收集的输出/日志文件
│   │   └── constraints.txt          # 可选的任务约束/说明
│   ├── verification/                # 验证与评分系统
│   │   ├── evaluator.py             # [核心] 评分脚本入口
│   │   ├── requirements.txt         # 运行评分环境所需的依赖
│   │   └── docker/                  # 环境容器化配置
│   │       └── Dockerfile           # 确保评测环境一致性
│   └── baseline/                    # [可选] 基础解法/示例代码
│       ├── solution.py              # 参考代码实现
│       └── result_log.txt           # 参考代码的运行日志或评分结果
└── <Task_Name_B>/                   # 该领域下的另一个任务
    └── ...
```
> 上述目录结构仅作为参考模板。在确保包含所有核心要素（如背景、输入输出、评测指标）的前提下，贡献者可根据具体情况调整文件组织方式。同时，验证代码的编程语言与格式均不作限制。
>
> 新增 benchmark 贡献默认必须通过 unified task 格式接入。也就是说，需要在 `<Task_Name>/frontier_eval/` 下补齐 benchmark 本地元数据，并使用 `task=unified` 完成框架适配验证。除非能够明确说明 unified 格式无法表达该任务、且已与维护者先沟通达成一致，否则不要再新增 `frontier_eval/tasks/<task>/...` 这种自定义 task 实现。完整 unified 元数据说明见 `frontier_eval/README_zh-CN.md`。

### 提交规范

1. 运行测试命令尽量简短（最好单行命令）提交前必须测试！
    1. python verification/evaluator.py scripts/init.py # 在benchmark下的运行，使用verification/evaluator.py作为评测入口，测试的目标也即agent evolve的目标为scripts/init.py
    2. python -m frontier_eval task=unified task.benchmark=<Domain_Name>/<Task_Name> algorithm.iterations=0 # 新增 benchmark 的框架适配验证。请在 README 中写清 unified benchmark id、任何必要的 runtime 覆盖项（例如 `task.runtime.conda_env=...`），以及 benchmark 自己需要的环境准备方式（额外 env、Docker、`third_party/`、自定义 `python_path` 等）
2. 请注意不要包含私人信息的文件，例如:.env、API keys、IDE 配置（.vscode/）、临时文件（*.log, temp/, __pycache__/）、个人测试脚本，同时请检查提交的内容中是否包含绝对路径，避免出现复现问题和个人隐私泄露。

3. **EVOLVE-BLOCK 标记（ShinkaEvolve / ABMCTS 必需）**：被 Agent 演化（evolve）的文件（例如 `scripts/init.py`，或类似 `malloclab-handout/mm.c` 这类语言特定的 baseline）必须包含 `EVOLVE-BLOCK-START` 与 `EVOLVE-BLOCK-END` 标记，用于定义*唯一*允许修改的代码区域。
   - 请保留标记行本身不变，并将标记之外的代码视为只读（CLI/I/O 契约、约束检查、评测器胶水代码等）。
   - 按语言使用正确的注释形式：
     - Python：`# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`
     - C/C++/CUDA/Rust/Swift：`// EVOLVE-BLOCK-START` / `// EVOLVE-BLOCK-END`

### 贡献流程

我们采用标准的 GitHub 协作流程：

1. **Fork 本仓库**: 点击右上角的 "Fork" 按钮，将项目复刻到你的 GitHub 账户。
2. **创建分支 (Branch)**:
* 在本地 Clone 你的 Fork 仓库。
* 创建一个新的分支进行开发，建议命名格式为：`feat/<Domain>/<TaskName>` (例如: `feat/Astrodynamics/MarsLanding`)。

3. **添加/修改内容**:
* 按照上述提交格式添加你的工程问题文件。
* 确保包含所有必要的说明文档和验证代码。

4. **本地测试**: 运行 `evaluator.py` 或构建 Docker 镜像，确保评测逻辑无误且能正常运行。
5. **提交 Pull Request (PR)**:
* 将修改 Push 到你的远程 Fork 仓库。
* 向本仓库的 `main` 分支发起 Pull Request。
* **PR 描述**: 请简要说明该 Task 的背景、来源以及如何运行验证代码。

6. **代码审查**: 
   * **Agent Review**: 提交 PR 后，首先由 **AI Agent** 进行自动化初步审查（包括代码规范、基础逻辑验证等），并可能在 PR 中直接提出修改建议。
   * **Maintainer Review**: Agent 审查通过后，**维护者** 将进行最终复核。确认无误后，你的贡献将被合并。
---
> 💡 如果这是你第一次贡献，或者对目录结构有疑问，欢迎先提交 Issue 进行讨论。

## 📊 任务进度与规划

完整的 benchmark 覆盖表与规划说明已移至 [TASK_PROGRESS_zh-CN.md](TASK_PROGRESS_zh-CN.md)。

该文件保留了完整任务列表、状态、贡献者 / 审查者信息，以及当前 `v1` 纳入范围说明，包括 `MuonTomography` 暂不计入有效任务池的说明。

## 📦 `best_code_only`

`best_code_only/` 位于仓库根目录，收录了当前已整理出的各实验 / 算法 / 模型 / task 对应的最终全局 best 代码。

目录结构：

```text
best_code_only/
└── <experiment>/
    └── <algorithm>/
        └── <model>/
            └── <task>/
                └── <code-file>
```

使用说明：

- 如果你要先确认某个实验、算法或模型是否补齐，先看 `best_code_only/coverage.json`。
- 直接进入对应 task 目录即可拿到该组合的最终 best 代码。示例路径：
  - `best_code_only/experiment1/openevolve/gpt-5.4/Astrodynamics_MannedLunarLanding/`
  - `best_code_only/experiment2/shinkaevolve/claude-opus-4.6/KernelEngineering_TriMul/`
- 文件名保留原始任务里的命名和后缀，因此可能是 `.py`、`.c`、`.cpp`，也可能是其他 task 自带文件名。

## 🧪 评测框架
初步实现部分评测算法与 benchmark 的对接。实现的核心部分见 `./frontier_eval`，使用方法详见[评测 README](frontier_eval/README_zh-CN.md)。注意：部分可选算法/任务依赖 `third_party/` 下的外部仓库（需要本地 clone），请按评测 README 的说明进行配置。

## 💬 加入社区

欢迎加入我们的开发者社区！无论你是想讨论新的工程问题构想、寻找任务合作者，还是在贡献过程中遇到了技术问题，都可以在群里与我们随时交流。

* 🟢 **飞书**: [点击这里加入我们的飞书讨论群](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=21ak5858-60ba-44fd-9085-01f165c8771c)
* 🔜 **Discord**: [点击这里加入我们的Discord社区](https://discord.gg/hxeVhZNN)

<!-- AI_GENERATED -->
