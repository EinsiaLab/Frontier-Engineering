# MicrowaveAbsorberDesign

[English](./README.md) | 简体中文

## 概览

该任务要求设计一个工作在 X 波段（8-12 GHz）的单层 PEC 背板吸波体。优化器需要选择吸波层厚度，以及基体、介电填料和磁性填料的体积分数，在吸收性能、厚度、密度和成本之间做折中。

## 快速开始

```bash
pip install -r verification/requirements.txt
python verification/evaluator.py scripts/init.py
python verification/evaluator.py baseline/solution.py
```

最终评分为 `combined_score`，由 evaluator 根据反射损耗曲线和工程 proxy 项统一计算。细节见 [Task.md](./Task.md)。
