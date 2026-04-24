# Torsion Profile Fitting

[English](./README.md) | 简体中文

## 概览

该 MolecularMechanics 任务要求针对目标 profile 数据拟合 torsion 参数。它是当前 v2 集合中三道 OpenFF 任务里最重的一题，并依赖 OpenFF runtime。

## 运行时

- 框架入口：`.venvs/frontier-v2-extra`
- benchmark runtime：`.venvs/openff-dev`

## Unified 运行

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=molecular_mechanics_torsion_profile_fitting \
  algorithm=openevolve \
  algorithm.iterations=0
```
