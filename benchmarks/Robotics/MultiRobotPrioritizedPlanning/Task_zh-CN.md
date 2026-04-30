# 多机器人优先级路径规划

## 任务概览

在一组多机器人栅格 case 上规划无碰撞路径集合，并尽量降低 hidden case 的平均总路径代价。

评测现在会使用多组 `public` / `hidden` 多机器人地图。每个 case 会固定栅格、机器人起点和目标点，而你需要返回整组路径。评分目标是总路径代价，同时报告 makespan 作为辅助诊断。

## 哪些部分是冻结的

- `runtime/problem.py` 中的 public 与 hidden 栅格 case。
- 每个 case 中固定的机器人起点/终点分配。
- 冲突规则：顶点冲突和对向换边冲突都视为非法。
- 总路径代价与 makespan 的定义。

## 提交接口

提交一个 Python 文件，定义：

```python
def plan_paths(grid, starts, goals):
    ...
```

返回每个机器人的路径列表；也接受带 `paths` 字段的字典。

## 评测流程

1. 载入每个 public / hidden case。
2. 对每个 case 调用 `plan_paths(grid, starts, goals)`。
3. 检查每个机器人的起终点、相邻移动、避障合法性，以及顶点/换边冲突。
4. 聚合不同 case 的总路径代价；最终分数使用 hidden 平均总成本。

## 指标

- `combined_score`：`-hidden_avg_total_cost`
- `valid`：只有所有 case 都返回合法无碰撞路径集时才为 `1.0`
- `public_avg_total_cost`
- `hidden_avg_total_cost`
- `baseline_hidden_avg_total_cost`
- `hidden_avg_makespan`
- `num_public_cases`
- `num_hidden_cases`

## 判为无效的情况

- 缺少 `plan_paths(...)`，或函数执行报错
- 返回值无法解析为多机器人路径集合
- 任意机器人路径起终点错误
- 任意路径包含非相邻移动或进入障碍物
- 出现顶点冲突或换边冲突
- 任意 public 或 hidden case 在评测中失败

<!-- AI_GENERATED -->
