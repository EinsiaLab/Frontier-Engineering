# Weighted Parameter Coverage

[English](./README.md) | 简体中文

## 概览

该 MolecularMechanics 任务要求在覆盖目标下选择力场参数。它属于当前 v2 任务集，但使用特殊的 OpenFF runtime，而不是纯 `uv` 环境。

## 运行时

- 框架入口：`.venvs/frontier-v2-extra` 或等价 `frontier_eval` 驱动环境
- benchmark runtime：`.venvs/openff-dev`

## Unified 运行

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=molecular_mechanics_weighted_parameter_coverage \
  algorithm=openevolve \
  algorithm.iterations=0
```

等价的显式 unified 命令：

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=MolecularMechanics/weighted_parameter_coverage \
  task.runtime.python_path=uv-env:openff-dev \
  algorithm=openevolve \
  algorithm.iterations=0
```
