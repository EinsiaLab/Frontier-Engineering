# DuckDB Pre-Aggregation Selection 任务

## 目标

Choose a small set of pre-aggregation tables for a frozen DuckDB reporting workload.

## 提交接口

提交一个 Python 文件，定义：

```python
def select_preaggregations(workload_manifest):
    ...
```

返回值必须是 whitelist 中的预聚合表名列表。也接受包含 `preaggregations` 字段的字典。

## 评测方式

评测器会：

1. 构建固定的 DuckDB workload。
2. 创建所选预聚合表。
3. 运行固定 reporting workload，并检查结果是否保持一致。
4. 记录候选总运行时间，并把 baseline 总运行时间作为诊断信息输出。

## 指标

- `combined_score`：`-candidate_total_runtime_s`
- `valid`：只有名称合法且结果保持不变时才为 `1.0`
- `candidate_total_runtime_s`
- `baseline_total_runtime_s`
- `candidate_setup_runtime_s`
- `candidate_workload_runtime_s`
