# DuckDB Index Selection 任务

## 目标

Choose a small set of DuckDB indexes for a frozen analytical lookup workload.

## 提交接口

提交一个 Python 文件，定义：

```python
def select_indexes(workload_manifest):
    ...
```

返回值必须是 whitelist 中的索引名列表。也接受包含 `indexes` 字段的字典。

## 评测方式

评测器会：

1. 构建固定的 DuckDB workload。
2. 创建所选索引。
3. 固定重复执行查询 workload 四次。
4. 记录候选总运行时间，并把无索引 baseline 作为诊断信息一并输出。

## 指标

- `combined_score`：`-candidate_total_runtime_s`
- `valid`：只有索引名合法且执行成功时才为 `1.0`
- `candidate_total_runtime_s`
- `baseline_total_runtime_s`
- `candidate_setup_runtime_s`
- `candidate_workload_runtime_s`
