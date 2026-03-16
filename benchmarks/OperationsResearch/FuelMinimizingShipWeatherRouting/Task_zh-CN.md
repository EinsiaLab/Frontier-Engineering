# Fuel-Minimizing Ship Weather Routing 任务

## 目标

Route a ship across a frozen coastal grid while minimizing total fuel consumption under synthetic wind and current fields.

## 提交接口

提交一个 Python 文件，定义：

```python
def solve(instance):
    ...
```

返回值可以是路径列表，也可以是包含 `path` 键的字典。

路径必须：

1. 从 `instance["start"]` 出发
2. 以 `instance["goal"]` 结束
3. 只能在相邻网格之间移动
4. 不能进入陆地或不可航行水域

## 固定世界模型

- 地图、起终点、合成风场与合成流场都固定在 `runtime/problem.py` 中。
- 上游算法谱系来自 `WeatherRoutingTool`，但这里的环境数据是 benchmark 内部固定生成的 synthetic asset。

## 评测方式

评测器会：

1. 载入固定实例
2. 机械检查路径可行性
3. 计算该路径的总燃油和总航时
4. 记录最短步数 baseline 与 Dijkstra 参考值作为诊断信息，同时直接以候选燃油目标打分

## 指标

- `combined_score`：`-candidate_fuel`
- `valid`：只有路径可行时才为 `1.0`
- `candidate_fuel`
- `baseline_fuel`
- `reference_fuel`
- `candidate_time_h`
- `baseline_time_h`
