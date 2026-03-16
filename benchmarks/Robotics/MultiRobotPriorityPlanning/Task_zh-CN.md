# Multi-Robot Priority Planning 任务

## 目标

Plan collision-free paths for three robots on a frozen occupancy grid while minimizing total path cost.

评测使用 `runtime/problem.py` 中冻结的一张多机器人 occupancy grid。

## 提交接口

提交一个 Python 文件，定义：

```python
def plan_paths(grid, starts, goals):
    ...
```

返回值必须是路径列表，每个机器人一条路径。也接受包含 `paths` 字段的字典。

## 指标

- `combined_score`：`-candidate_total_cost`
- `valid`：所有机器人路径都无碰撞时为 `1.0`
- `candidate_total_cost`
- `baseline_total_cost`
- `candidate_makespan`
- `lower_bound_total_cost`
