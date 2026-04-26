# NanoCarbonAbsorberOptimization

[English](./README.md) | 简体中文

## 任务特点

该任务是一个 **混合变量** 优化问题，用于优化 Nd-BaM 复合吸波材料中的纳米碳类型和含量，以提升 2–18 GHz 范围内的宽带吸波性能。

与当前材料域里纯连续变量的任务不同，这题同时包含：

- **离散变量**：碳材料类型（`CNTs` / `GO` / `OLC`）
- **连续变量**：碳含量（1-10%）和厚度（1.5-5 mm）

这更贴近真实工程决策：不仅要决定“多少”，还要决定“用哪一种材料”。

## 快速开始

```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```

## Unified 运行

```bash
bash scripts/run_v2_unified.sh MaterialEngineering/NanoCarbonAbsorberOptimization \
  algorithm=openevolve \
  algorithm.iterations=0
```
