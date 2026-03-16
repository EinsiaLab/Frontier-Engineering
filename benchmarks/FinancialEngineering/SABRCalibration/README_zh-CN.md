# SABR 波动率曲面校准

## 概述

将 SABR 随机波动率模型（Hagan et al. 2002）校准到市场隐含波动率曲面。
给定多个行权价和到期日的隐含波动率，找出能最优复现市场微笑的三个 SABR 参数（α, ρ, ν）。

本任务捕捉了量化金融核心问题——**波动率微笑**：不同行权价下市场期权价格隐含的 Black-Scholes
波动率不同，形成特征性的"微笑"或"偏斜"，简单的平坦波动率模型无法复现。

## 来源

| 字段 | 详情 |
|------|------|
| **Benchmark** | Hagan, Kumar, Lesniewski, Woodward (2002) "Managing Smile Risk", *Willmott Magazine* |
| **参考实现** | [QuantLib](https://github.com/lballabio/QuantLib) — C++ 量化金融库，**>4 000 ★**，LGPL 许可证 |
| **市场数据结构** | 代表性 SPX 期权波动率曲面（标准化远期价格 F=100）|
| **真实参数** | α=2.5, ρ=-0.65, ν=0.40, β=0.5（β 固定）|

## 问题背景

SABR 模型（Hagan 2002）通过随机微分方程描述随机波动率。通过解析近似，可以得到隐含波动率的
闭合解 σ_SABR(F, K, T; α, ρ, ν)（含 ATM 和非 ATM 两个分支）。

**校准目标**：最小化 RMSE = √( mean{ (σ_SABR(...) − σ_market)² } )

覆盖数据集中全部 33 个 (T, K) 对（3 个到期日 × 11 个行权价）。

## 题目难度

**3 / 5** — 中等难度

- 需正确实现 SABR 闭合公式（ATM 分支处理非平凡）。
- 目标函数非凸，初始化不当会陷入局部最小值。
- 从 ATM 波动率估计 α 的初步比例关系是关键洞察；偏斜和曲率需要正确的 rho/nu 估计。

## 输入 / 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| `F` | float | 标准化 ATM 远期价格（100.0）|
| `beta` | float | 固定 SABR beta = 0.5 |
| `T_list` | list[float] | 到期日 [0.25, 0.50, 1.00] 年 |
| `strikes` | list[int] | [70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120] |
| `market_vols` | list[list[float]] | 3×11 市场隐含波动率 |

**输出**：`[alpha, rho, nu]` — 三个浮点数，满足 `alpha>0`，`-1<rho<1`，`nu>0`。

## Validity（有效性）

输出 `[alpha, rho, nu]` 满足约束即为有效，评测器自动强制执行。

## 评分

```
combined_score = min(1.0, HUMAN_BEST_RMSE / RMSE)
HUMAN_BEST_RMSE = 0.001（通过正确的 SABR 校准可达）
```

- 基线（平坦 ATM 波动率，无偏斜）：RMSE ≈ 0.022 → 得分 ≈ **0.046**
- 良好校准的 SABR：RMSE < 0.001 → 得分 = **1.000**

## Human Best / 已知最优

使用 scipy.optimize 进行正确的 SABR 校准可恢复真实参数（α=2.5, ρ=-0.65, ν=0.40），精度达机器精度 → RMSE < 1e-6 → `combined_score = 1.0`。

参考文献：Hagan et al. (2002) 公式 (2.17b)。

## 运行方式

```bash
# 基线
python baseline/init.py

# 评测候选方案
python verification/evaluate.py baseline/init.py
```

## 依赖

```
numpy
scipy
```

无需 GPU、Docker 或外部数据下载。
