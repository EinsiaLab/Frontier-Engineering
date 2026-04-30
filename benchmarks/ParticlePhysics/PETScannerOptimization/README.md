# Particle Physics: PET Scanner Geometry and Cost Pareto Optimization

English | [简体中文](./README_zh-CN.md)

## Overview

This task optimizes the geometry of 20 PET detector rings under a strict crystal-volume budget. The agent must trade off photon sensitivity, parallax error, and material consumption.

## Local Run

```bash
pip install -r verification/requirements.txt
python baseline/solution.py
python verification/evaluator.py solution.json
```

The official baseline in this repository is the generated 20-ring `solution.py` output, with a verified score of about `598.1943`.

## Unified Run

```bash
bash scripts/run_v2_unified.sh ParticlePhysics/PETScannerOptimization \
  algorithm=openevolve \
  algorithm.iterations=0
```

Invalid submissions are rejected if they do not contain exactly 20 rings with unique contiguous `ring_id` values and bounded finite geometry variables.
