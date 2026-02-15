# calibration.md

本文件记录 `HighReliableSimulation` 的评测常量标定过程与冻结结果。

## 1. 目标

- 目标误码率：`BER ~= 1e-7`
- 固定配置：
  - `HammingCode(r=7, n=127)`
  - `decoder = chase(t=3)`
  - 评测接口：`MySampler.simulate_variance_controlled(...)`

## 2. 标定脚本

脚本位置：

- `benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/calibrate_sigma.py`

示例命令（发布预算）：

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/calibrate_sigma.py \
  --sigmas 0.264 0.268 0.272 0.276 \
  --target-ber 1e-7 \
  --target-std 0.05 \
  --max-samples 100000 \
  --batch-size 10000 \
  --min-errors 20 \
  --repeats 5 \
  --output benchmarks/WirelessChannelSimulation/HighReliableSimulation/calibration_final.json
```

输出文件：

- `benchmarks/WirelessChannelSimulation/HighReliableSimulation/calibration_final.json`

## 3. 冻结结果（当前生效）

- `sigma = 0.268`
- `target_std = 0.05`
- `max_samples = 100000`
- `batch_size = 10000`
- `min_errors = 20`
- `repeats = 5`
- 输出：`calibration_final.json`

当前已写入 `eval/evaluator.py` 的常量：

- `sigma = 0.268`
- `r0 = 5.52431776694918e-07`
- `r0_log = -14.408935892274092`
- `t0 = 0.18551087379455566`
- `target_std = 0.05`
- `max_samples = 100000`
- `batch_size = 10000`
- `min_errors = 20`
- `epsilon = 0.8`

当前观测（外层重复评测 `baseline/solution.py`）：

- `error_log_ratio` 均值约 `0.364`，标准差约 `0.276`
- `valid` 比例约 `5/8`
- 分数存在明显波动（重要性采样估计方差仍偏高）

稳定性调整：

- `epsilon` 从 `0.35` 放宽到 `0.8`。
- 在 `epsilon=0.8` 下，最近 5 次外层重复中 `baseline` 有效次数约 `4/5`，`init` 仍稳定为无效（0 分）。

## 4. 发布前检查

1. 用更高预算重跑标定（提高 `max_samples` / `repeats`）。
2. 确认 `sigma_star` 在 2 次独立标定中一致（或差异很小）。
3. 更新评测器常量并复测 `init.py`、`baseline/solution.py`。
4. 在 `Task.md` 和 `README.md` 同步最终冻结参数。
