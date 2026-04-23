# 粒子物理：PET 探测器几何与经济帕累托优化

[English](./README.md) | 简体中文

## 1. 任务简介

本任务（PET 探测器几何优化）是 `Frontier-Eng` 基准测试中**粒子物理与医疗工程**领域的一个顶级优化难题。

正电子发射断层扫描（PET）利用了反物质湮灭的物理现象——放射性示踪剂发射的正电子在与电子相遇时，质量完全转化为能量，释放出两束呈 180 度背对背的 511 keV 伽马射线。本任务要求 AI Agent 在极其严苛的经济与物理约束下，优化扫描仪探测环的几何尺寸。



> **核心挑战**：高能的 511 keV 伽马射线极难被阻挡，这需要又厚又贵的人造 LYSO 闪烁晶体。Agent 必须运用 3D 几何与指数衰减计算，在“最大化系统灵敏度”、“最小化空间视差误差（DOI 效应）”以及“确保总晶体体积不超预算”这三者之间，寻找一个极具挑战性的帕累托最优解。

关于为 Agent 设计的详细物理数学模型、目标函数以及输入输出格式，请参阅核心任务文档：[Task_zh-CN.md](./Task_zh-CN.md)。
关于物理鸿沟的学术文献支撑，请参阅：[reference/references.txt](./reference/references.txt)。

## 2. 本地运行

准备好 `frontier-eval-2` 环境后，你可以直接从任务目录运行基准测试：

```bash
conda activate frontier-eval-2
cd benchmarks/ParticlePhysics/PETScannerOptimization
python baseline/solution.py
python verification/evaluator.py solution.json
```

`verification/requirements.txt` 目前仅需要 `numpy>=1.24.0`。

上述基线已在本仓库中验证，结果如下：

```json
{"status": "success", "score": 73.80681701043795, "metrics": {"volume_mm3": 5089380.098815464, "sensitivity_factor": 0.1637684467863431, "resolution_gamma": 6.4031242374328485, "cost_penalty": 0.0}}
```

## 3. 使用 `frontier_eval` 运行

该任务在 `frontier_eval` 中注册为 `pet_scanner_optimization`。

在仓库根目录下，标准的兼容性检查命令为：

```bash
conda activate frontier-eval-2
python -m frontier_eval task=pet_scanner_optimization algorithm=openevolve algorithm.iterations=0
```

完成 [frontier_eval/README.md](../../../frontier_eval/README.md) 中描述的框架级 `.env` 或模型配置后，可以通过增加 `algorithm.iterations` 来启动真实的搜索，例如：

```bash
conda activate frontier-eval-2
python -m frontier_eval task=pet_scanner_optimization algorithm=openevolve algorithm.iterations=10
```

## 4. 评估指标 (Metrics)

`evaluator.py` 以标准 JSON 格式输出结果：
* `score`: 最终的综合得分（越高越好）。
* `metrics`: 包含内部细节，如 `volume_mm3`（消耗的晶体总体积）、`sensitivity_factor`（光子捕获灵敏度，越高越好）、`resolution_gamma`（空间分辨率退化程度，越低越好）以及 `cost_penalty`（超出体积预算的巨额惩罚）。