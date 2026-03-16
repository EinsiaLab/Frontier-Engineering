# Cantilever Compliance Topology Optimization 任务

## 目标

Minimize compliance on a frozen cantilever beam using pyMOTO's SIMP formulation and a fixed material budget.

评测在 `runtime/problem.py` 中冻结了一个基于 pyMOTO 的结构拓扑优化实例。

## 提交接口

提交一个 Python 文件，定义：

```python
def update_density(density, sensitivity, state):
    ...
```

输入参数：

- `density`：当前密度向量，NumPy 数组，形状为 `(nel,)`
- `sensitivity`：当前目标函数相对于设计变量的灵敏度
- `state`：字典，包含：
  - `iteration`
  - `domain_shape`
  - `volume_fraction`
  - `target_density_sum`
  - `minimum_density`
  - `move_limit`
  - `current_compliance`
  - `history`
  - `passive_solid_mask`
  - `passive_void_mask`

返回值必须是下一步的可行密度向量。也接受包含 `density` 字段的字典。

如果你只想先产生一个原始提案，可以从 `runtime.problem` 导入 `project_density`，把原始提案投影回可行域。

## 评测方式

评测器会：

1. 构建固定的 pyMOTO 有限元模型。
2. 运行固定 30 次优化迭代。
3. 对比 baseline 的 OC 更新规则与你的 `update_density(...)`。
4. 拒绝任何非有限或不可行的密度更新。
5. 直接以最终 candidate compliance 作为优化分数。

## 指标

- `combined_score`：`-candidate_compliance`
- `valid`：所有密度更新都有限且可行时为 `1.0`
- `candidate_compliance`
- `baseline_compliance`
- `final_volume_fraction`
- `volume_fraction_error`
