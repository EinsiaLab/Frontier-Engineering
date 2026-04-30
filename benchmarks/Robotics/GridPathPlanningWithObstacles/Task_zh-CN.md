# 带障碍栅格路径规划

## 任务概览

在一组二维占据栅格上规划无碰撞路径，并尽量降低 hidden case 的平均路径代价。

这个 benchmark 不再使用单张冻结地图。评测现在会运行多组 `public` / `hidden` 栅格，它们会改变走廊布局和障碍瓶颈。目标是在整组 case 上都返回合法且尽量短的路径。

## 哪些部分是冻结的

- `runtime/problem.py` 中的 public 与 hidden 栅格 case。
- 固定移动规则：每一步必须留在空闲区域内，并且只能在相邻格点之间移动。
- 固定路径代价定义：路径长度减一。

## 提交接口

提交一个 Python 文件，定义：

```python
def plan_path(grid, start, goal):
    ...
```

返回由 `(x, y)` 坐标组成的路径；也接受带 `path` 字段的字典。

## 评测流程

1. 载入每个 public / hidden 栅格 case。
2. 对每个 case 独立调用 `plan_path(grid, start, goal)`。
3. 检查起终点、相邻移动规则和避障合法性。
4. 聚合不同 case 的路径代价；最终分数使用 hidden 平均值。

## 指标

- `combined_score`：`-hidden_avg_cost`
- `valid`：只有所有 case 都返回合法无碰撞路径时才为 `1.0`
- `public_avg_cost`
- `hidden_avg_cost`
- `baseline_hidden_avg_cost`
- `num_public_cases`
- `num_hidden_cases`

## 判为无效的情况

- 缺少 `plan_path(...)`，或函数执行报错
- 返回值无法解析为路径
- 任意路径起终点错误
- 任意路径包含非相邻移动或进入障碍物
- 任意 public 或 hidden case 在评测中失败

<!-- AI_GENERATED -->
