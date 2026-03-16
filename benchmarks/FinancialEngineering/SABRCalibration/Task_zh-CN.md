# 任务：SABR 波动率曲面校准

## 任务名称

`sabr_calibration`

## 任务描述

实现 `solve(instance)` 函数，返回 SABR 模型参数 `[alpha, rho, nu]`，
使模型隐含波动率与市场观测波动率在 33 个（行权价, 到期日）组合上的 RMSE 最小。

SABR 模型（Hagan 2002）隐含波动率闭合近似公式：

**当 F ≠ K 时：**
```
σ(F,K,T) = α / { (FK)^((1-β)/2) · [1 + (1-β)²/24·ln²(F/K) + (1-β)⁴/1920·ln⁴(F/K)] }
           × z/χ(z) × [1 + ( (1-β)²α² / (24(FK)^(1-β))
                              + ρβνα / (4(FK)^((1-β)/2))
                              + (2-3ρ²)ν²/24 ) · T ]

其中 z = ν/α · (FK)^((1-β)/2) · ln(F/K)
     χ(z) = ln[ (√(1-2ρz+z²) + z - ρ) / (1-ρ) ]
```

**当 F = K（ATM）时：**
```
σ_ATM = α/F^(1-β) · [1 + ((1-β)²α²/(24F^(2-2β)) + ρβνα/(4F^(1-β)) + (2-3ρ²)ν²/24) · T]
```

## 接口

```python
def solve(instance: dict) -> list[float]:
    """
    参数：
        instance: {
            'F': float,           # ATM 远期价格（100.0）
            'beta': float,        # 固定为 0.5
            'T_list': list,       # [0.25, 0.50, 1.00] 年
            'strikes': list,      # [70,...,120]（11 个值）
            'market_vols': list,  # shape [3][11] 隐含波动率
        }
    返回：
        [alpha, rho, nu]，满足 alpha>0, -1<rho<1, nu>0
    """
```

## 评分

```
RMSE = sqrt( mean( (sabr_vol(F, K, T, α, ρ, ν) - market_vol[T,K])² ) )
combined_score = min(1.0, 0.001 / RMSE)
```

分数越高越好。当 RMSE ≤ 0.001 时，combined_score = 1.0。

## 基线得分

基线（从 ATM 波动率估计 α，ρ=0，ν=0.3）：得分 ≈ **0.046**

## Human Best 得分

正确校准的 SABR：得分 = **1.000**（RMSE < 1e-6）

## 约束

- `alpha > 0`
- `-1 < rho < 1`
- `nu > 0`
- `beta` 固定为 0.5（不得修改）
