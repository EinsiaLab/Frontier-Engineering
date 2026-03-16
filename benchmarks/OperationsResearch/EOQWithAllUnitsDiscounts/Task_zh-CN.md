# EOQ with All-Units Discounts 任务

## 目标

Choose an order quantity under piecewise all-units discount pricing.

规范来源来自 `Stockpyl` 的 all-units discount EOQ 实现。固定评测样例定义在 `runtime/problem.py` 中，属于 benchmark 内部冻结参数表。

## 提交接口

提交一个 Python 文件，定义：

```python
def solve(instance):
    ...
```

返回值要求：

- EOQ 类任务：返回 `order_quantity` 字段的字典，或者直接返回数值型订货批量。
- `(r,Q)` 类任务：返回包含 `reorder_point` 和 `order_quantity` 的字典，或者直接返回二元组 `(r, Q)`。

## 评测方式

评测器会：

1. 读取 `runtime/problem.py` 中的固定样例。
2. 运行 baseline。
3. 运行选手的 `solve(instance)`。
4. 计算成本和可行性。
5. 计算平均候选成本，并将其直接暴露为优化分数。

## 指标

- `combined_score`：`-avg_cost`
- `valid`：所有 case 都可行且数值有限时为 `1.0`
- `avg_cost`：平均候选成本
- `avg_cost_ratio`：仅用于诊断的平均 `baseline_cost / candidate_cost`
