# HighReliableSimulation 任务说明

## 目标

在稀有错误场景下，估计 AWGN 信道中 Hamming(127,120) 的 BER。
你需要实现 `MySampler`，并支持方差受控仿真。

## 提交协议

提交一个 Python 文件，需定义：

1. `class MySampler(SamplerBase)`
2. `MySampler.simulate_variance_controlled(...)`

评测器调用方式：

```python
sampler = MySampler(code=code, seed=seed)
result = sampler.simulate_variance_controlled(
    code=code,
    sigma=DEV_SIGMA,
    target_std=TARGET_STD,
    max_samples=MAX_SAMPLES,
    batch_size=BATCH_SIZE,
    fix_tx=True,
    min_errors=MIN_ERRORS,
)
```

其中 `code` 由评测器固定为 `HammingCode(r=7, decoder="binary")`，并设置 `ChaseDecoder(t=3)`。

## 返回格式

支持以下两类：

- 至少包含 6 项的 tuple/list：
  `(errors_log, weights_log, err_ratio, total_samples, actual_std, converged)`
- 具有等价字段的 dict。

评测器按 `err_rate_log = errors_log - weights_log` 解释误码率对数。

## 冻结评测常量

- `sigma = 0.268`
- `target_std = 0.05`
- `max_samples = 100000`
- `batch_size = 10000`
- `min_errors = 20`
- `r0 = 5.52431776694918e-07`
- `t0 = 0.18551087379455566`
- `epsilon = 0.8`
- `repeats = 3`

## 评分规则

- `e = |log(r / r0)|`，其中 `r = exp(err_rate_log)`。
- 若 `e >= epsilon`，得分为 `0`。
- 否则：`score = t0 / (t * e + 1e-6)`，其中 `t` 为运行时间中位数。

## 失败条件

以下任一情况得分为 `0`：

- `MySampler` 接口缺失或不合法
- 返回值不合法或出现非有限数值
- 运行失败
