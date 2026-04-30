# LightweightBroadbandAbsorber

[English](./README.md) | 简体中文

## 概览

该任务针对 8.2-18 GHz 频段的轻量宽带吸波体设计，要求在带宽、反射损耗、厚度、密度和成本之间取得平衡。

## 关键特征

- 4 种材料组分，存在性能与重量之间的竞争关系
- 存在最小 `EAB` 硬约束（`>= 4.0 GHz`）
- 密度惩罚是主导惩罚项

## 快速开始

```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```

## Unified 运行

```bash
bash scripts/run_v2_unified.sh MaterialEngineering/LightweightBroadbandAbsorber \
  algorithm=openevolve \
  algorithm.iterations=0
```
