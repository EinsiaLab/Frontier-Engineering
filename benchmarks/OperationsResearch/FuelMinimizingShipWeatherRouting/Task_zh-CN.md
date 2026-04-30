# 最小燃料天气航线规划

## 任务概览

在一组天气航线 case 上规划船舶路径，在满足最晚到达约束的前提下，尽量降低 hidden case 的平均燃料消耗。

评测现在会使用多组 `public` / `hidden` case，它们会改变风场、流场、海岸线和最晚到达预算。好的方法应当能在整组 case 上平衡燃料和航时，而不是只对单张冻结地图有效。

## 哪些部分是冻结的

- `runtime/problem.py` 中定义的 public 与 hidden 路由 case。
- 四邻接移动规则和陆地掩码。
- 固定的单步燃料/航时模型，包括迎风和流场影响。
- 每个 case 的最晚到达约束。

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
3. 检查返回航线的几何合法性和移动合法性。
4. 计算每个 case 的燃料和航时；未满足最晚到达的航线直接判失败。

## 指标

- `combined_score`：`-hidden_avg_fuel`
- `valid`：只有所有 case 都给出可行且准时的航线时才为 `1.0`
- `public_avg_fuel`
- `hidden_avg_fuel`
- `baseline_hidden_avg_fuel`
- `num_public_cases`
- `num_hidden_cases`

## 判为无效的情况

- 缺少 `solve(...)`，或函数执行报错
- 返回值无法解析为路径
- 任意航线进入陆地或包含非相邻移动
- 任意航线未满足最晚到达约束
- 任意 public 或 hidden case 在评测中失败

<!-- AI_GENERATED -->
