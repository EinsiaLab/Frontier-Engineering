# 库存优化任务总览

这个目录包含 5 个相关但不同的 Hard 模式任务，统一基于 stockpyl。

## 环境配置
```bash
pip install stockpyl numpy scipy
```

## 统一结构
每个任务都使用相同目录规范：
- `baseline/init.py`：简单基线实现（不使用优化器）
- `verification/reference.py`：基于 stockpyl 的参考实现
- `verification/evaluate.py`：统一调用两种算法并按同一评分函数对比
- `output/`：输出文件（`baseline_result.json`、`reference_result.json`、`comparison.json`）

## 任务之间的联系
- 都是在不确定需求/供给条件下做库存决策优化。
- 都用 0~1 加权评分，并比较 baseline 与 reference。
- 都刻意设置为 baseline 显著弱于 reference（便于评测区分）。

## 任务之间的差异
1. `tree_gsm_safety_stock`
   - 问题类型：树形多级安全库存配置
   - 关键模型：GSM（承诺服务时间）
2. `general_meio`
   - 问题类型：一般拓扑 MEIO（仿真目标）
   - 关键模型：非树网络的 base-stock 优化
3. `joint_replenishment`
   - 问题类型：多 SKU 联合补货
   - 关键模型：补货周期与倍数协同
4. `finite_horizon_dp`
   - 问题类型：有限期动态补货
   - 关键模型：时变 `(s_t, S_t)` 策略
5. `disruption_eoqd`
   - 问题类型：含供应中断的批量决策
   - 关键模型：EOQ with disruptions

## 任务难度说明
- `tree_gsm_safety_stock`：中等（2.5/5）
  - 难点：决策维度较小，但评分中同时要求 SLA 合规和策略复杂度控制。
- `disruption_eoqd`：中等偏上（3/5）
  - 难点：虽然是单变量决策，但要在成本、服务、风险、资金占用之间做随机环境下的权衡。
- `joint_replenishment`：中高（3.5/5）
  - 难点：离散变量（补货倍数）与连续变量（基准周期）联合优化，且要兼顾协同与响应速度。
- `finite_horizon_dp`：高（4/5）
  - 难点：时变策略设计 + 随机需求 + 仿真打分，需要动态规划与策略调参能力。
- `general_meio`：很高（4.5/5）
  - 难点：多级网络、仿真驱动目标、压力场景鲁棒性和节点服务平衡同时约束。

## 一键运行全部评估
在 `benchmarks/InventoryOptimization/` 目录下执行：

```bash
for d in tree_gsm_safety_stock general_meio joint_replenishment finite_horizon_dp disruption_eoqd; do
  echo "== $d =="
  python "$d/verification/evaluate.py"
done
```

## 接入 frontier_eval Unified Task
每个子任务都已在自身目录下提供 unified 元数据：

- `<子任务>/frontier_eval/initial_program.txt`
- `<子任务>/frontier_eval/candidate_destination.txt`
- `<子任务>/frontier_eval/eval_command.txt`
- `<子任务>/frontier_eval/agent_files.txt`
- `<子任务>/frontier_eval/readonly_files.txt`
- `<子任务>/frontier_eval/artifact_files.txt`
- `<子任务>/frontier_eval/constraints.txt`
- `<子任务>/frontier_eval/run_eval.py`

在仓库根目录运行（`.venvs/frontier-eval-driver` 作为驱动环境，`.venvs/frontier-v1-main` 作为任务运行环境）：

```bash
.venvs/frontier-eval-driver/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=InventoryOptimization/tree_gsm_safety_stock \
  task.runtime.env_name=frontier-v1-main \
  algorithm=openevolve \
  algorithm.iterations=0
```

运行全部 5 个子任务：

```bash
for d in tree_gsm_safety_stock general_meio joint_replenishment finite_horizon_dp disruption_eoqd; do
  .venvs/frontier-eval-driver/bin/python -m frontier_eval \
    task=unified \
    task.benchmark=InventoryOptimization/$d \
    task.runtime.env_name=frontier-v1-main \
    algorithm=openevolve \
    algorithm.iterations=0
done
```

## 运行时长说明
以下为本地 `frontier-v1-main` 运行时单次评测耗时（近似值）：

- `tree_gsm_safety_stock`：约 3-4 秒
- `general_meio`：约 20-25 秒（耗时较长，仿真较重）
- `joint_replenishment`：约 1 秒
- `finite_horizon_dp`：约 6 秒
- `disruption_eoqd`：约 3 秒
