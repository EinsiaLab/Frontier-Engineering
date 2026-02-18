# StructuralOptimization Benchmarks 运行指南

本目录包含两个结构优化基准测试：ISCSO2015 和 ISCSO2023。

## 快速运行

### ISCSO 2015 (45杆 2D 桁架)

```bash
cd benchmarks/StructuralOptimization/ISCSO2015

# 1. 运行基线解法（生成 submission.json）
python3 baseline/solution.py

# 2. 评估提交
python3 verification/evaluator.py baseline/solution.py

# 或直接评估已有的 submission.json
python3 verification/evaluator.py --test submission.json
```

**预期输出**：
- `submission.json`: 包含 54 个设计变量（45 个截面积 + 9 个节点 y 坐标）
- 评估结果：结构重量（kg），可行解应 < 400 kg

### ISCSO 2023 (284杆 3D 桁架)

```bash
cd benchmarks/StructuralOptimization/ISCSO2023

# 1. 运行基线解法
python3 baseline/solution.py

# 2. 评估提交
python3 verification/evaluator.py baseline/solution.py

# 或直接评估已有的 submission.json
python3 verification/evaluator.py --test submission.json
```

**预期输出**：
- `submission.json`: 包含 284 个截面积设计变量
- 评估结果：结构重量（kg），可行解应 < 8000 kg

## 通过 frontier_eval 框架运行

```bash
# 从项目根目录运行
python3 -m frontier_eval.tasks.iscso2015.task
python3 -m frontier_eval.tasks.iscso2023.task
```

## 文件结构

每个 benchmark 包含：
- `Task.md` / `Task_zh-CN.md`: 问题规范
- `references/problem_data.json`: 问题数据
- `verification/evaluator.py`: 评估脚本（核心）
- `verification/fem_truss*.py`: FEM 求解器
- `baseline/solution.py`: 基线解法
- `scripts/init.py`: OpenEvolve 演化入口

## 依赖

- Python 3.8+
- numpy
- scipy (ISCSO2023 需要)

```bash
pip install numpy scipy
```

## 注意事项

1. `submission.json` 会在运行 `baseline/solution.py` 时生成，位于当前工作目录
2. 评估器会在临时目录中运行候选程序，确保路径正确
3. 两个 benchmark 都是最小化问题，`combined_score = -weight`（越小越好 → 分数越大）

