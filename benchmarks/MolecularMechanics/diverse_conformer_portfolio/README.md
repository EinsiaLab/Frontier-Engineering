# Diverse Conformer Portfolio

English | [简体中文](./README_zh-CN.md)

## Overview

This MolecularMechanics task builds a conformer portfolio balancing low energy and structural diversity. It is part of the current v2 task set and runs with the OpenFF runtime.

## Runtime

- framework entrypoint: `.venvs/frontier-v2-extra`
- benchmark runtime: `.venvs/openff-dev`

## Unified Run

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=molecular_mechanics_diverse_conformer_portfolio \
  algorithm=openevolve \
  algorithm.iterations=0
```
