# Diverse Conformer Portfolio

[English](./README.md) | 简体中文

## 概览

该 MolecularMechanics 任务要求构建一个在低能量与结构多样性之间折中的构象组合。它属于当前 v2 任务集，运行时依赖 OpenFF 环境。

## 运行时

- 框架入口：`.venvs/frontier-v2-extra`
- benchmark runtime：`.venvs/openff-dev`

## Unified 运行

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=molecular_mechanics_diverse_conformer_portfolio \
  algorithm=openevolve \
  algorithm.iterations=0
```
