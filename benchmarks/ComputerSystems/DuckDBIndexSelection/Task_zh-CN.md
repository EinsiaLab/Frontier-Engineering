# DuckDB 索引选择

## 任务概览

在一组分析型 DuckDB workload 上，从白名单中选择少量索引，尽量降低 hidden case 的平均运行时间。

这个 benchmark 不再是单一冻结 workload。评测会在 `runtime/problem.py` 中定义的多组 `public` 与 `hidden` manifest 上运行，它们会改变查询混合、时间过滤和 lookup 强度。好的策略应当能在多组 manifest 上稳定工作，而不是只对某一个 case 调参。

## 哪些部分是冻结的

- `benchmarks/ComputerSystems/duckdb_local_workload.py` 中的本地 DuckDB schema 与数据生成逻辑。
- 每个 workload manifest 中给出的合法索引白名单，以及该 case 的索引预算上限。
- 固定计时协议：创建所选索引，先做一次 warm-up，再对该 case 重复执行 workload 并计时。

## 提交接口

提交一个 Python 文件，定义：

```python
def select_indexes(workload_manifest):
    ...
```

返回索引名列表；也接受带 `indexes` 字段的字典。

## 评测流程

1. 从 `runtime/problem.py` 读取 `PUBLIC_CASES` 与 `HIDDEN_CASES`。
2. 对每个 case，把 case-specific manifest 传入 `select_indexes(...)`。
3. 创建候选索引并运行该 case workload，测量总耗时。
4. 分别聚合 public 与 hidden 耗时；最终分数使用 hidden 平均耗时。

## 指标

- `combined_score`：`-hidden_avg_runtime_s`
- `valid`：只有所有 case 都成功执行且索引名全部合法时才为 `1.0`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`
- `num_public_cases`
- `num_hidden_cases`

## 判为无效的情况

- 缺少 `select_indexes(...)`，或函数执行报错
- 返回值无法解析为索引名列表
- 任意索引名不在白名单中
- 任意 case 超过该 case 的索引预算
- 任意 public 或 hidden case 在建索引或执行 workload 时失败

<!-- AI_GENERATED -->
