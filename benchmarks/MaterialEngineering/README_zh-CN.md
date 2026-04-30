# 材料工程

[English](./README.md) | 简体中文

## 领域背景

本仓库中的材料工程任务关注物理性能、厚度、密度和制造成本之间的显式工程折中，同时保持 unified 本地评测可运行。

在当前分支上，材料工程任务统一通过 benchmark-local `task=unified` 元数据接入，并默认使用 `.venvs/frontier-v2-extra` 作为轻量运行环境。

## 任务索引

* **[微波吸波材料设计](./MicrowaveAbsorberDesign/README.md)**
  * **背景**：单层 X 波段 PEC 背板吸波体设计。
  * **目标**：优化厚度和组分比例，在反射损耗、有效带宽、密度和成本之间取得平衡。
* **[轻量宽带吸波材料设计](./LightweightBroadbandAbsorber/README.md)**
  * **背景**：面向 8.2-18 GHz 的轻量宽带吸波体设计。
  * **目标**：在带宽、吸收深度、厚度、密度和成本之间折中，并满足最小 EAB 硬约束。
* **[纳米碳吸波材料优化](./NanoCarbonAbsorberOptimization/README.md)**
  * **背景**：在 Nd-BaM 复合体系中联合选择纳米碳类型和含量。
  * **目标**：在 2-18 GHz 频段内优化离散碳材料选择、碳含量和厚度。
