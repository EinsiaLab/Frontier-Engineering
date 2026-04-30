# NanoCarbonAbsorberOptimization — 任务说明

## 1. 背景

纳米碳材料的类型与含量会显著影响铁氧体基复合材料的吸波性能。碳纳米管（CNTs）、氧化石墨烯（GO）和洋葱状碳（OLC）分别对应不同的介电损耗机制。

本 benchmark 基于 Nd₀.₁₅-BaM/NC 复合体系，目标频段为 **2–18 GHz**。任务是一个 **混合变量优化** 问题：既要选择最合适的碳材料类型（离散变量），又要联合优化碳含量和吸波层厚度（连续变量）。

## 2. 设计变量

| 变量 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `carbon_type` | 离散 | `"CNTs"`, `"GO"`, `"OLC"` | 纳米碳材料类型 |
| `carbon_content` | 连续 | `[0.01, 0.10]` | 纳米碳质量分数 |
| `d_mm` | 连续 | `[1.5, 5.0]` mm | 吸波层厚度 |

## 3. 评估方式

### 3.1 有效参数模型

复合材料的有效电磁参数取决于所选碳材料类型和含量：

`eps_eff = eps_base + slope * carbon_content`

具体参数见 `references/material_db.json`。

### 3.2 指标

- 频率范围：2.0–18.0 GHz（321 个采样点）
- `RL_min`：最小反射损耗
- `EAB_10`：满足 `RL <= -10 dB` 的最大连续带宽

### 3.3 硬约束

若 `EAB_10 < 3.0 GHz`，则判为 infeasible，`combined_score = 0`。

### 3.4 最终得分

所有指标先做 `[0, 1]` 归一化，再按如下方式组合：

`combined_score = reward(EAB_10, |RL_min|) - penalty(thickness, density, cost)`

实际以 `verification/evaluator.py` 的实现为准。

## 4. 输出约定

候选程序必须写出 `temp/submission.json`：

```json
{
  "benchmark_id": "nanocarbon_absorber_optimization_2_18ghz",
  "carbon_type": "CNTs",
  "carbon_content": 0.04,
  "d_mm": 1.5
}
```

## 5. 判无效条件

以下情况会被判为无效：

- 输出缺失或格式错误
- 必需字段缺失
- `benchmark_id` 不匹配
- `carbon_type` 不在 `"CNTs"`、`"GO"`、`"OLC"` 之中
- `carbon_content` 非有限值或超出 `[0.01, 0.10]`
- `d_mm` 非有限值或超出 `[1.5, 5.0]`
- `EAB_10 < 3.0 GHz`
- 候选程序超时或非零退出
