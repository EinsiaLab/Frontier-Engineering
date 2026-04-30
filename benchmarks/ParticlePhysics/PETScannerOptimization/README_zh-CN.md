# 粒子物理：PET 探测器几何与经济帕累托优化

[English](./README.md) | 简体中文

## 概览

该任务要求在严格晶体体积预算下优化 20 个 PET 探测环的几何参数，在光子灵敏度、视差误差和材料消耗之间做折中。

## 本地运行

```bash
pip install -r verification/requirements.txt
python baseline/solution.py
python verification/evaluator.py solution.json
```

本仓库中的官方 baseline 为 `solution.py` 生成的 20 环设计，验证分数约为 `598.1943`。

## Unified 运行

```bash
bash scripts/run_v2_unified.sh ParticlePhysics/PETScannerOptimization \
  algorithm=openevolve \
  algorithm.iterations=0
```

若提交不是恰好 20 个 ring，或 `ring_id` 不唯一/不连续，或几何参数越界，将被直接判为无效。
