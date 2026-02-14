# Frontier-Eng: 人工智能代理的大规模工程优化基准

[English](README.md) | 简体中文

**Frontier-Eng** 是一个旨在评估 AI Agent 在**真实工程领域**中解决**开放式优化问题**能力的Benchmark。

不同于现有的关注计算机科学（CS）或纯数学抽象问题的 Benchmark，Frontier-Eng 聚焦于具有实际**经济效益**和**物理约束**的工程难题，预期涵盖航天、土木、EDA、生物工程等多个领域。

## 🎯 动机

当前的 AI4Research 评测体系存在以下局限性：

1. **评估方式单一**：大多采用 0/1 二元评估或封闭区间的 Rubric，无法有效衡量 Agent 在开放世界中通过交互进行**迭代优化**的能力。
2. **领域局限**：现有 Benchmark 大多局限于 CS 领域（如代码生成），或将实际问题高度抽象为数学题，剥离了现实世界的复杂性，使得 Agent 无法利用丰富的外部知识和工具。
3. **指标偏差**：传统计算指标关注模型的平均表现，而对于工程优化问题，我们更应关注模型在单一问题上通过探索机制所能达到的**极值（Peak Performance）**。

**Frontier-Eng** 旨在通过提供丰富的上下文和工具支持，评估 Agent 在广泛工程学科中解决具有实际价值问题的能力。

## 🤝 贡献指南

我们需要社区的力量来扩展 Benchmark 的覆盖范围。我们欢迎通过 Pull Request (PR) 的方式提交新的工程问题。如果你希望贡献，请遵循以下标准和流程：
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

下表列出了当前 Benchmark 中各领域任务的覆盖情况。我们不仅欢迎代码贡献，也欢迎社区提出具有挑战性的新工程问题构想。
| 领域 | 任务名称 | 状态 | 维护者/贡献者 | 备注 |
| :--- | :--- | :---: | :--- | :--- |
| **Astrodynamics** | `MannedLunarLanding` | 已完成 | @jdp22 | 登月软着陆轨迹优化 |
| **ElectronicDesignAutomation** | `IntegrationPhysicalDesignOptimization` | 开发中 | @ahydchh | 芯片宏单元布局优化 |

> 💡 **有新的工程问题想法？**
> 即使你暂时无法提供完整的验证代码，我们也非常欢迎你分享好的**Task 构想**！
> 请创建一个 Issue 详细描述该问题的**现实背景**与**工程价值**。经讨论确认后，我们会将其加入上表，集结社区力量共同攻克。

## 💬 加入社区

欢迎加入我们的开发者社区！无论你是想讨论新的工程问题构想、寻找任务合作者，还是在贡献过程中遇到了技术问题，都可以在群里与我们随时交流。

* 🟢 **飞书**: [点击这里加入我们的飞书讨论群](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=a1cuff9f-347a-43ce-8825-79c2a38038c6)
* 🔜 **Discord / Slack**: (筹备中，即将推出...)