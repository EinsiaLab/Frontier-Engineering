# 参与贡献 Frontier-Eng

[English](CONTRIBUTING.md)

我们需要社区的力量来扩展 Benchmark 的覆盖范围。我们欢迎通过 Pull Request (PR) 的方式提交新的工程问题。如果你希望贡献，请遵循以下标准和流程。

> **AI 辅助贡献**：我们欢迎使用 AI 工具辅助创建的贡献。若使用 agent，请让其读取 **`skill/source/frontier-contributor/`**（见其中 `SKILL.md`）。**但是，请不要过度依赖 AI 工具或完全放手不管**。人工审查和监督对于确保质量和正确性至关重要。

## 样本要求

1. **Reality Gap**: 必须贴近现实，考虑现实影响因素，非单纯数学抽象。
2. **Economic Value**: 问题解决后应具有明确的工程或经济价值。
3. **Verifiability**: 必须提供可执行的验证程序（Docker 优先），能在可接受时间内完成评测。

## 提交格式

每一个 Task 应当包含以下文件结构：

```text
<Domain_Name>/                       # 一级目录：领域名称 (e.g., Astrodynamics)
├── README.md                        # [必选] 领域综述 (默认入口，中英文均可)：介绍背景及子任务索引
├── README_zh-CN.md                  # [可选] 领域综述 (中文版。仅当 README.md 为英文且提供了中文版时使用)
├── <Task_Name_A>/                   # 二级目录：具体任务名称 (e.g., MannedLunarLanding)
│   ├── README.md                    # [必选] 导航文档：说明文件结构、如何运行及快速开始
│   ├── README_zh-CN.md              # [可选] 导航文档
│   ├── Task.md                      # [必选] 任务详情文档：核心文档，包含背景、物理模型、输入输出定义
│   ├── Task_zh-CN.md                # [可选] 任务详情文档
│   ├── references/                  # 参考资料目录
│   │   ├── constants.json           # 物理常数、仿真参数等
│   │   └── manuals.pdf              # 领域知识手册、物理方程或约束条件文档
│   ├── frontier_eval/               # [必选] Frontier Eval 的 unified-task 元数据
│   │   ├── initial_program.txt      # 初始可编辑程序路径（相对任务根目录）
│   │   ├── eval_command.txt         # `task=unified` 使用的评测命令模板
│   │   ├── agent_files.txt          # 提供给 Agent 的上下文文件
│   │   ├── artifact_files.txt       # 评测后需要收集的输出/日志文件
│   │   └── constraints.txt          # 可选的任务约束/说明
│   ├── verification/                # 验证与评分系统
│   │   ├── evaluator.py             # [核心] 评分脚本入口
│   │   ├── requirements.txt         # 运行评分环境所需的依赖
│   │   └── docker/                  # 环境容器化配置
│   │       └── Dockerfile           # 确保评测环境一致性
│   └── baseline/                    # [可选] 基础解法/示例代码
│       ├── solution.py              # 参考代码实现
│       └── result_log.txt           # 参考代码的运行日志或评分结果
└── <Task_Name_B>/                   # 该领域下的另一个任务
    └── ...
```

> 上述目录结构仅作为参考模板。在确保包含所有核心要素（如背景、输入输出、评测指标）的前提下，贡献者可根据具体情况调整文件组织方式。同时，验证代码的编程语言与格式均不作限制。
>
> 新增 benchmark 贡献默认必须通过 unified task 格式接入。也就是说，需要在 `<Task_Name>/frontier_eval/` 下补齐 benchmark 本地元数据，并使用 `task=unified` 完成框架适配验证。除非能够明确说明 unified 格式无法表达该任务、且已与维护者先沟通达成一致，否则不要再新增 `frontier_eval/tasks/<task>/...` 这种自定义 task 实现。完整 unified 元数据说明见 `frontier_eval/README_zh-CN.md`。

## 提交规范

1. 运行测试命令尽量简短（最好单行命令），提交前必须测试！

   1. `python verification/evaluator.py scripts/init.py` — 在 benchmark 下运行，使用 `verification/evaluator.py` 作为评测入口；测试目标也即 agent evolve 的目标为 `scripts/init.py`。
   2. `python -m frontier_eval task=unified task.benchmark=<Domain_Name>/<Task_Name> algorithm.iterations=0` — 新增 benchmark 的框架适配验证。请在 README 中写清 unified benchmark id、任何必要的 runtime 覆盖项（例如 `task.runtime.conda_env=...`），以及 benchmark 自己需要的环境准备方式（额外 env、Docker、`third_party/`、自定义 `python_path` 等）。

2. 请注意不要包含私人信息的文件，例如：`.env`、API keys、IDE 配置（`.vscode/`）、临时文件（`*.log`、`temp/`、`__pycache__/`）、个人测试脚本；同时请检查提交的内容中是否包含绝对路径，避免出现复现问题和个人隐私泄露。

3. **EVOLVE-BLOCK 标记（ShinkaEvolve / ABMCTS 必需）**：被 Agent 演化（evolve）的文件（例如 `scripts/init.py`，或类似 `malloclab-handout/mm.c` 这类语言特定的 baseline）必须包含 `EVOLVE-BLOCK-START` 与 `EVOLVE-BLOCK-END` 标记，用于定义*唯一*允许修改的代码区域。
   - 请保留标记行本身不变，并将标记之外的代码视为只读（CLI/I/O 契约、约束检查、评测器胶水代码等）。
   - 按语言使用正确的注释形式：
     - Python：`# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`
     - C/C++/CUDA/Rust/Swift：`// EVOLVE-BLOCK-START` / `// EVOLVE-BLOCK-END`

## 贡献流程

我们采用标准的 GitHub 协作流程：

1. **Fork 本仓库**：点击右上角的 "Fork" 按钮，将项目复刻到你的 GitHub 账户。
2. **创建分支 (Branch)**：
   - 在本地 Clone 你的 Fork 仓库。
   - 创建一个新的分支进行开发，建议命名格式为：`feat/<Domain>/<TaskName>`（例如：`feat/Astrodynamics/MarsLanding`）。
3. **添加/修改内容**：
   - 按照上述提交格式添加你的工程问题文件。
   - 确保包含所有必要的说明文档和验证代码。
4. **本地测试**：运行 `evaluator.py` 或构建 Docker 镜像，确保评测逻辑无误且能正常运行。
5. **提交 Pull Request (PR)**：
   - 将修改 Push 到你的远程 Fork 仓库。
   - 向本仓库的 `main` 分支发起 Pull Request。
   - **PR 描述**：请简要说明该 Task 的背景、来源以及如何运行验证代码。
6. **代码审查**：
   - **Agent Review**：提交 PR 后，首先由 **AI Agent** 进行自动化初步审查（包括代码规范、基础逻辑验证等），并可能在 PR 中直接提出修改建议。
   - **Maintainer Review**：Agent 审查通过后，**维护者** 将进行最终复核。确认无误后，你的贡献将被合并。

---

> 如果这是你第一次贡献，或者对目录结构有疑问，欢迎先提交 Issue 进行讨论。
