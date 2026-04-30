# LightweightBroadbandAbsorber — 任务说明

## 1. 背景

轻量宽带微波吸收材料对于航空航天、无人机和便携电子系统都很重要，因为这些场景同时要求电磁隐身与减重。本 benchmark 基于 CNTs@Nd-BaM/PE 复合体系，重点引入了宽带硬约束和更强的密度惩罚。

## 2. 设计变量

优化器控制 5 个变量，涉及 4 种材料组分：

- `d_mm`：厚度，范围 `[1.0, 5.0]`
- `phi_magnetic_absorber`：磁性吸收剂体积分数
- `phi_conductive_filler`：导电填料体积分数
- `phi_lightweight_magnetic`：轻量磁性组分体积分数
- `phi_matrix`：基体体积分数

约束：

- 所有体积分数和为 `1.0`
- 容差 `1e-6`

## 3. 评估方式

评测器使用线性体积分数混合规则计算等效电磁参数，并通过 PEC 背板传输线理论计算反射损耗曲线。

主要指标：

- `RL_min`
- `EAB_10`

硬约束：

- 若 `EAB_10 < 4.0 GHz`，则判为 infeasible，`combined_score = 0`

最终分数综合考虑：

- 带宽奖励
- 吸收深度奖励
- 厚度惩罚
- 密度惩罚
- 成本惩罚

实际以 `verification/evaluator.py` 为准。

## 4. 输出格式

候选程序必须写出 `temp/submission.json`，包含：

```json
{
  "benchmark_id": "lightweight_broadband_absorber_8_18ghz",
  "d_mm": 1.9,
  "phi_magnetic_absorber": 0.25,
  "phi_conductive_filler": 0.10,
  "phi_lightweight_magnetic": 0.05,
  "phi_matrix": 0.60
}
```

## 5. 判无效条件

以下情况会被判无效：

- 输出缺失或格式错误
- 必需字段缺失
- `benchmark_id` 不匹配
- 任意数值非有限或超出范围
- 体积分数之和不满足约束
- `EAB_10 < 4.0 GHz`
- 候选程序超时或非零退出
