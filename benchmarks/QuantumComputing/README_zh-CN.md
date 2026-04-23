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

当前 baseline 策略：
- `task_01`：先做 local rewrite，再做面向 target 的多 seed transpile 搜索。
- `task_02`：`local rewrite -> clifford+t transpile(opt=3) -> local rewrite`。
- `task_03`：按后端注册 equivalence，并使用 target-aware transpile 参数。

## 统一目录结构
每个题目都采用同一结构：
- `baseline/solve.py`：包含各题目当前 baseline 策略的 evolve 入口。
- `baseline/structural_optimizer.py`：由 `solve.py` 复用的 task-local local-rewrite 辅助模块。
- `verification/evaluate.py`：单一评测入口，同时包含 candidate 与 `opt0..opt3` 参考对比。
- `verification/utils.py`：公共工具函数。
- `tests/case_*.json`：多个有差异的测例。
- `README*.md` 与 `TASK*.md`：运行说明与任务定义。

## 评测命令
```bash
.venvs/frontier-eval-driver/bin/python -m frontier_eval task=unified task.benchmark=QuantumComputing/task_01_routing_qftentangled task.runtime.env_name=frontier-v1-main algorithm=openevolve algorithm.iterations=0
.venvs/frontier-eval-driver/bin/python -m frontier_eval task=unified task.benchmark=QuantumComputing/task_02_clifford_t_synthesis task.runtime.env_name=frontier-v1-main algorithm=openevolve algorithm.iterations=0
.venvs/frontier-eval-driver/bin/python -m frontier_eval task=unified task.benchmark=QuantumComputing/task_03_cross_target_qaoa task.runtime.env_name=frontier-v1-main algorithm=openevolve algorithm.iterations=0
```
