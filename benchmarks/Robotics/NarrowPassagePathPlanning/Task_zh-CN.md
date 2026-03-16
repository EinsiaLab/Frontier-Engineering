# Narrow Passage Path Planning 任务

## 目标

Plan a collision-free path through a single-cell narrow passage on a frozen occupancy grid.

评测使用 `runtime/problem.py` 中冻结的一张 occupancy grid。

## 提交接口

提交一个 Python 文件，定义：

```python
def plan_path(grid, start, goal):
    ...
```

返回值必须是一条 `(x, y)` 路径序列。也接受包含 `path` 字段的字典。

## 指标

- `combined_score`：`-candidate_cost`
- `valid`：路径有限且无碰撞时为 `1.0`
- `candidate_cost`
- `baseline_cost`
- `reference_cost`
