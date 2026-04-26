# Torsion Profile Fitting

English | [简体中文](./README_zh-CN.md)

## Overview

This MolecularMechanics task fits torsion parameters against target profile data. It is the heaviest of the three OpenFF tasks in the current v2 set and uses the OpenFF runtime.

## Runtime

- framework entrypoint: `.venvs/frontier-v2-extra`
- benchmark runtime: `.venvs/openff-dev`

## Unified Run

```bash
.venvs/frontier-v2-extra/bin/python -m frontier_eval \
  task=molecular_mechanics_torsion_profile_fitting \
  algorithm=openevolve \
  algorithm.iterations=0
```
