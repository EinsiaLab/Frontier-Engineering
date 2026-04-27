# DuckDB SQL 改写

## 任务概览

对一组分析型 SQL 进行改写，在保持结果完全等价的前提下，尽量降低 hidden case 的平均运行时间。

这个任务不再是单条冻结 SQL。评测现在会使用多组 `public` / `hidden` SQL case，它们会改变分组键、过滤条件和 rollup 形式。好的策略应当既保证语义完全一致，又能对整个 query family 带来稳定收益。

## 哪些部分是冻结的

- `benchmarks/ComputerSystems/duckdb_local_workload.py` 中的本地 DuckDB schema 与数据生成逻辑。
- `PUBLIC_CASES` 与 `HIDDEN_CASES` 中保存的 case-specific baseline SQL。
- 固定语义校验：候选结果必须与 baseline 查询结果逐行等价，浮点值只允许很小容差。

## 提交接口

提交一个 Python 文件，定义：

```python
def rewrite_query(sql, workload_manifest):
    ...
```

返回改写后的 SQL 字符串；runtime helper 也接受带 `sql` 字段的字典。

## 评测流程

1. 对每个 public / hidden case，把 baseline SQL 和 case manifest 传入 `rewrite_query(...)`。
2. 在全新的 DuckDB 数据库上分别执行 baseline SQL 与 candidate SQL。
3. 如果任意 case 的结果与 baseline 不一致，则直接判失败。
4. 聚合整个 case family 的运行时间；最终分数使用 hidden case 平均耗时。

## 指标

- `combined_score`：`-hidden_avg_runtime_s`
- `valid`：只有所有改写结果都语义等价且全部 case 成功运行时才为 `1.0`
- `public_avg_runtime_s`
- `hidden_avg_runtime_s`
- `baseline_hidden_avg_runtime_s`
- `num_public_cases`
- `num_hidden_cases`

## 判为无效的情况

- 缺少 `rewrite_query(...)`，或函数执行报错
- 返回值无法解释为 SQL
- 任意 public 或 hidden case 改变了查询结果
- 任意改写 SQL 执行失败
- 任意运行时间指标变成非有限值

<!-- AI_GENERATED -->
