# Agent-Evolve 量子优化题目集

本目录包含 3 个基于本仓库 `mqt.bench` API 构建的优化题目。

## 环境
请使用指定解释器：

```bash
pip install mqt.bench
```

## 题目列表
- `task_01_routing_qftentangled`：面向 IBM Falcon 的 mapped-level 路由优化。
- `task_02_clifford_t_synthesis`：面向 `clifford+t` 原生门集的综合优化。
- `task_03_cross_target_qaoa`：同一策略在 IBM 与 IonQ 双目标上的鲁棒优化。

## 统一目录结构
每个题目都采用同一结构：
- `baseline/solve.py`：evolve 入口。
- `baseline/structural_optimizer.py`：当前弱基线实现。
- `verification/evaluate.py`：单一评测入口，同时包含 candidate 与 `opt0..opt3` 参考对比。
- `verification/utils.py`：公共工具函数。
- `tests/case_*.json`：多个有差异的测例。
- `README*.md` 与 `TASK*.md`：运行说明与任务定义。

## 评测命令
```
python -m frontier_eval task=unified task.benchmark=QuantumComputing task_01_routing_qftentangled/task_02_clifford_t_synthesis/task_03_cross_target_qaoa task.runtime.conda_env=mqt algorithm=openevolve algorithm.iterations=0
```
