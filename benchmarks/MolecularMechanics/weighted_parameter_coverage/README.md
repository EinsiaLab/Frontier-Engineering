# Weighted Parameter Coverage

English | [简体中文](./README_zh-CN.md)

## Overview

This MolecularMechanics task selects force-field parameters under a coverage objective. It is part of the current v2 task set and uses the special OpenFF runtime rather than a pure `uv` environment.

## Runtime

- framework entrypoint: `.venvs/frontier-v2-extra` or equivalent `frontier_eval` driver runtime
- benchmark runtime: `.venvs/openff-dev`

## Unified Run

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=molecular_mechanics_weighted_parameter_coverage \
  algorithm=openevolve \
  algorithm.iterations=0
```

Equivalent explicit unified path:

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=unified \
  task.benchmark=MolecularMechanics/weighted_parameter_coverage \
  task.runtime.python_path=uv-env:openff-dev \
  algorithm=openevolve \
  algorithm.iterations=0
```
