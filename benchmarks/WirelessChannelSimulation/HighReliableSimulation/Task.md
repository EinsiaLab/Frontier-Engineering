# HighReliableSimulation Task

## 1. 任务目标

实现一个高效、稳定的采样器 `MySampler`（继承 `SamplerBase`），用于在固定参数下估计
Hamming(127,120) 码在 AWGN 信道中的 BER。

## 2. 固定配置

- 码参数：`r=7`（码长 `n=127`）
- 译码器：`chase(t=3)`（固定，不允许修改）
- 当前评测噪声：`sigma=0.268`（已冻结）
- 评测入口：固定调用 `MySampler.simulate_variance_controlled(...)`

## 3. 提交接口

提交 Python 文件必须包含：

1. `class MySampler(SamplerBase)`；
2. 方法 `simulate_variance_controlled(...)`，该方法内部执行仿真并返回结果。

推荐返回格式与 `HammingCode.simulate_variance_controlled` 保持一致：

`(errors_log, weights_log, err_ratio, total_samples, actual_std, converged)`

## 4. 评分

设：

- `r`：选手估计 BER
- `r0`：参考 BER（离线标定常量）
- `t`：选手耗时
- `t0`：基线耗时（常量）
- `e = |log(r / r0)|`

则：

- 有效性：`e < epsilon`
- 综合分数：`combined_score = t0 / (t * e + 1e-6)`（越大越好）

若接口错误、返回值非法、数值非有限或评测失败，则记 0 分。

## 5. 当前评测常量

- `sigma = 0.268`
- `target_std = 0.05`
- `max_samples = 100000`
- `batch_size = 10000`
- `min_errors = 20`
- `r0 = 5.52431776694918e-07`
- `t0 = 0.18551087379455566`
- `epsilon = 0.8`
