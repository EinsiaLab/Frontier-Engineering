# 粒子物理：调强质子治疗剂量权重优化

[English](./README.md) | 简体中文

## 1. 任务简介

本任务（Proton Therapy Planning Optimization）是 `Frontier-Eng` 基准测试在**粒子物理与医疗工程**领域的顶级优化问题。

质子治疗利用高能质子束独特的“布拉格峰（Bragg Peak）”物理特性——在穿透组织前段释放极少能量，而在到达特定深度时瞬间释放出绝大部分能量——来实现对肿瘤的“定向爆破”。本任务要求 AI Agent 在极其严苛的医疗安全约束下，优化质子笔形束（Pencil Beams）的三维空间停靠点和照射权重。

> **核心挑战**：肿瘤（CTV）通常紧贴着极度敏感的健康器官（OAR，如脑干）。Agent 必须通过精准的三维剂量核函数叠加计算，在“最大化肿瘤处方剂量覆盖率”和“确保脑干受照剂量不超标”之间寻找极具挑战性的帕累托最优解。

详细的物理数学模型、目标函数以及输入输出格式，请参阅给 Agent 阅读的专用说明文档：[Task_zh-CN.md](./Task_zh-CN.md)。

## 2. 本地运行 (Local Run)

在当前 v2 任务集中，本题的直接本地运行环境为 `.venvs/frontier-v2-extra`：

```bash
cd benchmarks/ParticlePhysics/ProtonTherapyPlanning
../../../.venvs/frontier-v2-extra/bin/python baseline/solution.py
../../../.venvs/frontier-v2-extra/bin/python verification/evaluator.py plan.json
```

`verification/requirements.txt` 目前仅依赖 `numpy>=1.24.0`。

上述基线代码已在本仓库中验证，并输出以下结果：

```json
{"score": -2685.8873258471367, "status": "success", "metrics": {"ctv_mse": 2779.3623258471366, "oar_overdose_penalty": 0.0, "total_weight": 130.5}}
```

## 3. 使用 `frontier_eval` 运行

本题现在已经通过 benchmark-local `task=unified` 元数据接入主线 v2 工作流。

在仓库根目录下，标准兼容性检查命令为：

```bash
bash scripts/run_v2_unified.sh ParticlePhysics/ProtonTherapyPlanning \
  algorithm=openevolve \
  algorithm.iterations=0
```

如果需要运行等价的显式 `frontier_eval` 命令：

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=ParticlePhysics/ProtonTherapyPlanning \
  algorithm=openevolve \
  algorithm.iterations=0
```

## 4. 评估指标

`evaluator.py` 会将结果输出为标准的 JSON 格式：
* `score`: 最终的综合得分（越大越好）。
* `metrics`: 包含内部明细，如 `ctv_mse`（肿瘤剂量均方误差，越小越好）、`oar_overdose_penalty`（健康器官过量惩罚）和 `total_weight`（总束流消耗）。
