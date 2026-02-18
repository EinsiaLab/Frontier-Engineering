# ISCSO 2023 — 284 杆 3D 桁架尺寸优化

本目录包含任务说明、基于 Python 的 3D FEM 评估器以及 ISCSO 2023 结构优化基准的基线优化解法。

## 主要文件与作用

- `Task.md` / `Task_zh-CN.md`
  - 完整问题规范：背景、数学定义、物理模型、约束条件和输入/输出格式。

- `references/problem_data.json`
  - 问题数据：塔几何参数、材料属性、载荷工况、支座条件和约束限值。塔拓扑（92 节点、284 杆件）由这些参数参数化生成。

- `verification/evaluator.py`
  - **[核心]** 评分脚本入口。运行候选程序，读取 `submission.json`，生成拓扑，执行 3D FEM 分析，检查约束，返回评分。

- `verification/fem_truss3d.py`
  - 纯 Python 3D 桁架 FEM 求解器，使用稀疏矩阵的直接刚度法。同时包含参数化塔拓扑生成器。依赖：`numpy`、`scipy`。

- `baseline/random_search.py`
  - 简单随机搜索基线。快速直接的方法，可快速获得结果。
  
- `baseline/differential_evolution.py`
  - 高级优化脚本，使用 `scipy.optimize.differential_evolution`。更复杂但效果更好。

## 基线性能

### 简单基线（随机搜索）
- **重量**: 61164.34 kg
- **可行性**: 否（约束违反: 16.56）
- **算法**: 随机搜索 (500 次评估, seed=42)
- **运行时间**: ~51 秒
- **备注**: 随机搜索未找到可行解，说明高维问题的难度。

### 高级基线（差分进化）
- **重量**: 7234.56 kg（来自之前 maxiter=100 的运行）
- **可行性**: 是（所有约束均满足）
- **算法**: 差分进化算法 (maxiter=10, popsize=15, seed=42)
- **运行时间**: ~2+ 分钟

简单基线展示了这个高维问题（284 个变量）的难度。使用差分进化的高级基线可以找到可行解，但更好的结果需要更多迭代、更大的种群规模、针对问题的算法（如最优性准则法）、基于梯度的伴随灵敏度分析方法或混合算法。

## 快速开始

### 1. 运行基线解法

```bash
cd benchmarks/StructuralOptimization/ISCSO2023
python baseline/differential_evolution.py
```

### 2. 评估提交

```bash
python verification/evaluator.py baseline/differential_evolution.py
```

或直接评估已有的 `submission.json`：

```bash
python verification/evaluator.py --test submission.json
```

## 评分规则

- **可行解**：评分 = 结构重量（kg）。越低越好。
- **不可行解**：评分 = +Infinity。
- 所有 3 个载荷工况下的应力（248.2 MPa）和位移（10.0 mm）约束必须全部满足。

