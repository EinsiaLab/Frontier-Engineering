# 动态流场最短航时船舶路径规划

## 任务概览

在一组沿海栅格 case 上规划船舶航线，在满足流场与吃水约束的前提下，尽量降低 hidden case 的平均航时。

这个 benchmark 不再是单张冻结地图。评测现在会使用多组 `public` / `hidden` 航线 case，它们会改变海岸线、流场分区、浅水格点和起终点位置。好的策略应当能在这组 case 上泛化，而不是记住某一条固定路线。

## 哪些部分是冻结的

- `runtime/problem.py` 中定义的 public 与 hidden 路由 case。
- 四邻接移动规则、最小水深约束和 hop 预算。
- 由确定性流场和水深场决定的固定航时计算方式。

## 提交接口

提交一个 Python 文件，定义：

```python
def solve(instance):
    ...
```

返回路径坐标列表，或带 `path` 字段的字典。

## 评测流程

1. 从 `runtime/problem.py` 载入每个 public / hidden case。
2. 对每个 case 独立调用 `solve(instance)`。
3. 检查路径起终点、相邻移动规则、可航行性和 hop 预算。
4. 计算每个 case 的航时，并分别聚合 public 与 hidden 平均值。

## 指标

- `combined_score`：`-hidden_avg_time_h`
- `valid`：只有所有 case 都给出可行航线时才为 `1.0`
- `public_avg_time_h`
- `hidden_avg_time_h`
- `baseline_hidden_avg_time_h`
- `num_public_cases`
- `num_hidden_cases`

## 判为无效的情况

- 缺少 `solve(...)`，或函数执行报错
- 返回值无法解析为路径
- 任意路径起终点错误
- 任意路径包含非相邻移动、进入陆地、违反最小水深约束或超过 hop 预算
- 任意 public 或 hidden case 在评测中失败

<!-- AI_GENERATED -->
