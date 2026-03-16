# FT10 Neighborhood Move Selection 任务

## 目标

Guide an adjacent-swap local search on the canonical FT10 Fisher-Thompson 10x10 job shop instance.

评测使用单个固定的 canonical 实例：`ft10`。
该实例的已知最优 makespan 为 `930`。

## 提交接口

提交一个 Python 文件。

如果是 dispatch-rule 任务，需要定义：

```python
def score_operation(operation, state):
    ...
```

如果是邻域搜索任务，需要定义：

```python
def score_move(move, state):
    ...
```

你也可以额外定义：

```python
MAX_ITERATIONS = 50
```

## 评测方式

Dispatch-rule 任务：

1. 从空排程开始。
2. 每次收集每个 job 的下一道未排工序。
3. 在“最早可开工时间最小”的工序集合中，选择 `score_operation` 最高者。
4. 构造完整可行排程并计算 makespan。

邻域搜索任务：

1. 从 baseline 的 SPT dispatch 排程开始。
2. 生成机器序列上的相邻交换 move。
3. 用 `score_move` 对 move 排序。
4. 按排序顺序找到第一个真正改进 makespan 的 move 并应用。
5. 当没有改进 move 或达到 `MAX_ITERATIONS` 时停止。

## 指标

- `combined_score`：`-candidate_makespan`
- `valid`：只有生成完整可行排程时才为 `1.0`
- `candidate_makespan`
- `baseline_makespan`
- `relative_gap_to_optimum`

## 失败情况

如果出现以下情况，提交会被判为无效，并得到一个很低的分数：

- 缺少要求的评分函数
- 返回值不是有限标量
- 诱导出的排程不可行
- 候选程序在评测时崩溃
