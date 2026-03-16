# DuckDB Query Rewrite 任务

## 目标

Rewrite a frozen DuckDB analytical SQL query to preserve results while reducing total runtime.

## 提交接口

提交一个 Python 文件，定义：

```python
def rewrite_query(sql, workload_manifest):
    ...
```

返回值必须是重写后的 SQL 字符串。也接受包含 `sql` 字段的字典。

## 评测方式

评测器会：

1. 构建固定的 DuckDB workload。
2. 执行原始 SQL，得到参考结果。
3. 执行候选重写 SQL，并严格检查结果等价。
4. 多次计时候选查询，并把 baseline 重写的运行时间作为诊断信息输出。

## 指标

- `combined_score`：`-candidate_runtime_s`
- `valid`：只有重写 SQL 保持结果不变时才为 `1.0`
- `candidate_runtime_s`
- `baseline_runtime_s`
- `row_count`
