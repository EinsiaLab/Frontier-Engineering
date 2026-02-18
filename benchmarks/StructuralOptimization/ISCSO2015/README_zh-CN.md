# ISCSO 2015 — 45 杆 2D 桁架尺寸 + 形状优化

本目录包含任务说明、基于 Python 的 FEM 评估器以及 ISCSO 2015 结构优化基准的基线优化解法。

## 主要文件与作用

- `Task.md` / `Task_zh-CN.md`
  - 完整问题规范：背景、数学定义、物理模型、约束条件和输入/输出格式。

- `references/problem_data.json`
  - 完整问题数据：节点坐标、杆件连接、材料属性、载荷工况、支座条件和约束限值。

- `verification/evaluator.py`
  - **[核心]** 评分脚本入口。运行候选程序，读取 `submission.json`，执行 FEM 分析，检查约束，返回评分。

- `verification/fem_truss2d.py`
  - 纯 Python 2D 桁架 FEM 求解器，使用直接刚度法。依赖：仅 `numpy`。

- `verification/requirements.txt`
  - 评测环境的 Python 依赖。

- `verification/docker/Dockerfile`
  - 容器化评测环境，确保可复现性。

- `baseline/differential_evolution.py`
  - 参考优化脚本，使用 `scipy.optimize.differential_evolution`。输出 `submission.json`。

## 基线性能

使用 `scipy.optimize.differential_evolution` 的基线解法达到：
- **重量**: 342.59 kg
- **可行性**: 是（所有约束均满足）
- **算法**: 差分进化算法 (maxiter=200, popsize=30, seed=42)

这提供了一个参考基准。通过更先进的算法、更大的优化预算或针对问题的启发式方法可以获得更好的结果。

## 快速开始

### 1. 运行基线解法

```bash
cd benchmarks/StructuralOptimization/ISCSO2015
python baseline/differential_evolution.py
```

在当前目录生成 `submission.json`。

### 2. 评估提交

```bash
python verification/evaluator.py baseline/differential_evolution.py
```

或直接评估已有的 `submission.json`：

```bash
python verification/evaluator.py --test submission.json
```

### 3. 使用 Docker 运行

```bash
cd verification/docker
docker build -t iscso2015-eval .
docker run -v $(pwd)/../../:/workspace iscso2015-eval python /workspace/baseline/differential_evolution.py
```

## 常见流程

1. 编写或修改优化脚本，输出 `submission.json`。
2. 运行评估器检查可行性和评分。
3. 迭代优化以降低目标值（最小化重量），同时保持可行性。

## 评分规则

- **可行解**：评分 = 结构重量（kg）。越低越好。
- **不可行解**：评分 = +Infinity。
- 所有载荷工况下的应力和位移约束必须全部满足。

