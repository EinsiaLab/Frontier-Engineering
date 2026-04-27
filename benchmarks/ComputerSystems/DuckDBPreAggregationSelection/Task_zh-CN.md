# DuckDB 预聚合选择

## 任务概览

在一组分析型 DuckDB 报表 workload 上，从白名单中选择少量预聚合结构，尽量降低 hidden case 的平均运行时间。

评测不再是单一冻结 workload，而是多组 `public` / `hidden` 报表配置。不同 case 会改变 segment 过滤、时间窗口、top-k 数量或报表重心。目标是在不改变语义的前提下，选择对多组 case 都有帮助的预聚合。

## 哪些部分是冻结的

- `benchmarks/ComputerSystems/duckdb_local_workload.py` 中的本地 DuckDB schema 与数据生成逻辑。
- 合法预聚合名称白名单，以及每个 case 的预聚合预算上限。
- 固定的语义校验：候选预聚合不能改变冻结报表族的输出结果。

## 提交接口

提交一个 Python 文件，定义：

```python
def select_preaggregations(workload_manifest):
    ...
```

返回预聚合名称列表；也接受带 `preaggregations` 字段的字典。

## 评测流程

1. 从 `PUBLIC_CASES` 与 `HIDDEN_CASES` 载入 case manifest。
2. 对每个 case，把该 case 的 manifest 传给 `select_preaggregations(...)`。
3. 物化所选预聚合，并验证报表输出语义不变。
4. 聚合不同 case 的运行时间；最终分数使用 hidden case 平均耗时。

## 指标

- `combined_score`：`-hidden_avg_runtime_s`
- `valid`：只有所有报表语义正确且所有 case 都成功运行时才为 `1.0`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`
- `num_public_cases`
- `num_hidden_cases`

## 判为无效的情况

- 缺少 `select_preaggregations(...)`，或函数执行报错
- 返回值无法解析为名称列表
- 任意名称不在白名单中
- 任意 case 超过预聚合预算
- 任意预聚合方案改变了报表结果
- 任意 public 或 hidden case 在构建或评测时失败

<!-- AI_GENERATED -->
