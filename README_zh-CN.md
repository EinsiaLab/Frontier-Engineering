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

> **AI 辅助贡献**：我们欢迎使用 AI 工具辅助创建的贡献。如果您使用 AI 助手来帮助完成贡献，我们建议将本仓库中的提示词指南（`AGENT.md` 或 `AGENT_zh-CN.md`）提供给您的 AI 助手，以确保其遵循我们的标准和要求。**但是，请不要过度依赖 AI 工具或完全放手不管**。人工审查和监督对于确保质量和正确性至关重要。
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

### 提交规范

1. 运行测试命令尽量简短（最好单行命令）提交前必须测试！
    1. python verification/evaluator.py scripts/init.py # 在benchmark下的运行，使用verification/evaluator.py作为评测入口，测试的目标也即agent evolve的目标为scripts/init.py
    2. python -m frontier_eval task=<task_name> algorithm.iterations=0 # 与框架的适配验证。注意，请在README中说明任务注册的task_name
2. 请注意不要包含私人信息的文件，例如:.env、API keys、IDE 配置（.vscode/）、临时文件（*.log, temp/, __pycache__/）、个人测试脚本，同时请检查提交的内容中是否包含绝对路径，避免出现复现问题和个人隐私泄露。

3. **单文件闭包（Baseline，必需）**：`scripts/init.py`（以及可选的 `baseline/solution.py`）必须自包含，便于 OpenEvolve 等算法进行单文件优化。
   - 不要 `import` 本仓库 `benchmarks/` 下的其他 Python 代码（例如任务目录下的其它 `.py` 文件）。
   - 允许导入 Python 标准库和 `verification/requirements.txt` 中声明的第三方依赖。

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

<table>
  <thead>
    <tr>
      <th>领域</th>
      <th>任务名称</th>
      <th>状态</th>
      <th>贡献者</th>
      <th>审查者</th>
      <th>备注</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Astrodynamics</b></td>
      <td><code>MannedLunarLanding</code></td>
      <td>已完成</td>
      <td>@jdp22</td>
      <td>@jdp22</td>
      <td>登月软着陆轨迹优化</td>
    </tr>
    <tr>
      <td><b>ElectronicDesignAutomation</b></td>
      <td><code>IntegrationPhysicalDesignOptimization</code></td>
      <td>开发中</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>芯片宏单元布局优化</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Kernel Engineering</b></td>
      <td><code>MLA</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>GPUMode MLA 解码内核</td>
    </tr>
    <tr>
      <td><code>TriMul</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>GPUMode 三角乘法</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Single Cell Analysis</b></td>
      <td><code>denoising</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Open Problems 单细胞分析</td>
    </tr>
    <tr>
      <td><code>perturbation_prediction</code></td>
      <td>已完成</td>
      <td>@llltttwww</td>
      <td>@llltttwww</td>
      <td>OpenProblems 扰动响应预测（NeurIPS 2023 scPerturb）</td>
    </tr>
    <tr>
      <td><code>predict_modality</code></td>
      <td>已完成</td>
      <td>@llltttwww</td>
      <td>@llltttwww</td>
      <td>OpenProblems 模态预测（NeurIPS 2021，RNA→ADT）</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Cryptographic</b></td>
      <td><code>AES-128 CTR</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Advanced Encryption Standard, 128-bit key, Counter mode</td>
    </tr>
    <tr>
      <td><code>SHA-256</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Secure Hash Algorithm 256-bit</td>
    </tr>
    <tr>
      <td><code>SHA3-256</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>Secure Hash Algorithm 3 256-bit</td>
    </tr>
    <tr>
      <td><b>Computer Systems</b></td>
      <td><code>Malloc Lab</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td>动态内存分配实验</td>
    </tr>
    <tr>
      <td><b>EngDesign</b></td>
      <td><code>CY_03, WJ_01, XY_05, AM_02, AM_03, YJ_02, YJ_03</code></td>
      <td>已完成</td>
      <td>@ahydchh</td>
      <td>@ahydchh</td>
      <td><a href="https://github.com/AGI4Engineering/EngDesign.git">EngDesign</a></td>
    </tr>
    <tr>
      <td rowspan="2"><b>StructuralOptimization</b></td>
      <td><code>ISCSO2015</code></td>
      <td>已完成</td>
      <td>@yks23</td>
      <td>@yks23</td>
      <td>45 杆 2D 桁架尺寸+形状优化</td>
    </tr>
    <tr>
      <td><code>ISCSO2023</code></td>
      <td>已完成</td>
      <td>@yks23</td>
      <td>@yks23</td>
      <td>284 杆 3D 桁架尺寸优化</td>
    </tr>
    <tr>
      <td><b>Aerodynamics</b></td>
      <td><code>CarAerodynamicsSensing</code></td>
      <td>已完成</td>
      <td>@LeiDQ, @llltttwww</td>
      <td>@llltttwww</td>
      <td>3D 汽车表面传感器布局优化，用于压力场重建</td>
    </tr>
    <tr>
      <td><b>WirelessChannelSimulation</b></td>
      <td><code>HighReliableSimulation</code></td>
      <td>已完成</td>
      <td>@tonyhaohan</td>
      <td>@yks23, @ahydchh</td>
      <td>使用重要性采样估计 Hamming(127,120) 的误码率</td>
    </tr>
  </tbody>
</table>
> 💡 **有新的工程问题想法？**
> 即使你暂时无法提供完整的验证代码，我们也非常欢迎你分享好的**Task 构想**！
> 请创建一个 Issue 详细描述该问题的**现实背景**与**工程价值**。经讨论确认后，我们会将其加入上表，集结社区力量共同攻克。

## 🧪 评测框架
初步实现部分评测算法与 benchmark 的对接。实现的核心部分见 `./frontier_eval`，使用方法详见[评测 README](frontier_eval/README_zh-CN.md)。

## 💬 加入社区

欢迎加入我们的开发者社区！无论你是想讨论新的工程问题构想、寻找任务合作者，还是在贡献过程中遇到了技术问题，都可以在群里与我们随时交流。

* 🟢 **飞书**: [点击这里加入我们的飞书讨论群](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=a1cuff9f-347a-43ce-8825-79c2a38038c6)
* 🔜 **Discord / Slack**: (筹备中，即将推出...)
