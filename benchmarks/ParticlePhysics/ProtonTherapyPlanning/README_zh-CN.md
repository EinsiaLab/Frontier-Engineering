# 粒子物理：调强质子治疗剂量权重优化

[English](./README.md) | 简体中文

## 1. 任务简介

本任务（Proton Therapy Planning Optimization）是 `Frontier-Eng` 基准测试在**粒子物理与医疗工程**领域的顶级优化问题。

质子治疗利用高能质子束独特的“布拉格峰（Bragg Peak）”物理特性——在穿透组织前段释放极少能量，而在到达特定深度时瞬间释放出绝大部分能量——来实现对肿瘤的“定向爆破”。本任务要求 AI Agent 在极其严苛的医疗安全约束下，优化质子笔形束（Pencil Beams）的三维空间停靠点和照射权重。

> **核心挑战**：肿瘤（CTV）通常紧贴着极度敏感的健康器官（OAR，如脑干）。Agent 必须通过精准的三维剂量核函数叠加计算，在“最大化肿瘤处方剂量覆盖率”和“确保脑干受照剂量不超标”之间寻找极具挑战性的帕累托最优解。

详细的物理数学模型、目标函数以及输入输出格式，请参阅给 Agent 阅读的专用说明文档：[Task_zh-CN.md](./Task_zh-CN.md)。

## 2. 文件结构

```text
ProtonTherapyPlanning/
├── README.md                        # 本导航文档的英文版
├── README_zh-CN.md                  # 本导航文档（中文版）
├── Task.md                          # [核心] Agent 任务说明与物理模型（英文）
├── Task_zh-CN.md                    # [核心] Agent 任务说明与物理模型（中文）
├── references/                      # 领域参考资料
│   └── constants.json               # 处方剂量、器官坐标等常数配置
├── verification/                    # 验证与评分系统
│   ├── evaluator.py                 # 核心评分 Python 脚本（计算三维剂量叠加）
│   ├── requirements.txt             # 评测环境依赖 (numpy)
│   └── docker/                      
│       └── Dockerfile               # 评测环境容器化配置
└── baseline/                        # 基线方案与参考代码
    └── solution.py                  # 生成初始基线解的参考代码
```

## 3. 快速开始 (Quick Start)

你可以通过本地 Python 环境或 Docker 来运行本任务的验证脚本。验证脚本会读取包含质子束坐标与权重的 JSON 文件，并在三维网格上计算剂量，最终输出评分。

### 方式一：本地 Python 运行

1. **安装依赖**：
   确保你的环境中安装了 `numpy`。
   ```bash
   cd verification
   pip install -r requirements.txt
   ```

2. **生成基线解答（可选）**：
   运行基线代码生成一个模拟的 Agent 输出文件 `plan.json`。
   ```bash
   cd ../baseline
   python solution.py
   ```

3. **运行评分脚本**：
   将生成的 JSON 文件路径传递给评分器。
   ```bash
   cd ../verification
   python evaluator.py ../baseline/plan.json
   ```

### 方式二：使用 Docker 运行 (推荐)

为了保证评测环境的绝对一致性，推荐使用 Docker 运行验证流程。

1. **构建镜像**：
   在 `verification/docker` 目录下执行构建命令。
   ```bash
   cd verification/docker
   docker build -t frontier-proton-eval -f Dockerfile ..
   ```

2. **运行容器进行评测**：
   将本地的解答文件挂载到容器中进行评测。
   ```bash
   docker run --rm -v $(pwd)/../../baseline/plan.json:/app/plan.json frontier-proton-eval /app/plan.json
   ```

## 4. 评估指标

`evaluator.py` 会将结果输出为标准的 JSON 格式：
* `score`: 最终的综合得分（越大越好）。
* `metrics`: 包含内部明细，如 `ctv_mse`（肿瘤剂量均方误差，越小越好）、`oar_overdose_penalty`（健康器官过量惩罚）和 `total_weight`（总束流消耗）。