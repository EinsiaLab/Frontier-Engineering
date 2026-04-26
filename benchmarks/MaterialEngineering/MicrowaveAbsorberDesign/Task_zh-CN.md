# MicrowaveAbsorberDesign — 任务说明

## 1. 背景

微波吸收材料在电磁兼容、雷达散射截面降低和电子设备屏蔽中都很重要。本 benchmark 聚焦于 **X 波段（8-12 GHz）** 的单层吸波体，并假设其背后为理想导体（PEC）。

## 2. 设计变量

优化器需要控制以下变量：

- `d_mm`：吸波层厚度，单位 mm，范围 `[1.0, 5.0]`
- `phi_dielectric`：介电填料体积分数，范围 `[0, 1]`
- `phi_magnetic`：磁性填料体积分数，范围 `[0, 1]`
- `phi_matrix`：基体体积分数，范围 `[0, 1]`

约束：

- `phi_dielectric + phi_magnetic + phi_matrix = 1.0`
- 容差为 `1e-6`

## 3. 评分方式

评测器先通过线性体积分数混合规则计算等效电磁参数，再在固定 X 波段频率网格上计算反射损耗曲线。

主要指标：

- `RL_min`：频带内最小反射损耗
- `EAB_10`：满足 `RL <= -10 dB` 的最大连续带宽

辅助工程 proxy：

- 等效密度
- 成本 proxy

最终标量目标为：

`combined_score = reward(EAB_10, |RL_min|) - penalty(thickness, density, cost)`

归一化范围和权重由 `references/problem_config.json` 给出；实际以 `verification/evaluator.py` 的实现为准。

## 4. 输出约定

候选程序必须写出 `temp/submission.json`，格式如下：

```json
{
  "benchmark_id": "microwave_absorber_single_layer_xband",
  "d_mm": 2.5,
  "phi_dielectric": 0.20,
  "phi_magnetic": 0.35,
  "phi_matrix": 0.45
}
```

## 5. 判无效条件

以下情况会被判为无效：

- 输出 JSON 缺失或格式错误
- 必需字段缺失
- `benchmark_id` 不匹配
- 任意值不是有限数或超出范围
- 三个体积分数之和不满足约束
- 候选程序超时或非零退出
