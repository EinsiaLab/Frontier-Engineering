# HighReliableSimulation 任务说明

本文件为任务核心文档，定义背景、物理/统计模型、提交接口、输入输出协议、评分指标与失败条件。

## 1. 背景与目标

在高可靠通信场景中，目标误码率（BER, Bit Error Rate）可能低至 1e-6 甚至更低。此时朴素蒙特卡洛往往难以在有限时间内观测到足够的错误事件，导致估计方差过大。

本任务要求参赛者实现一个高效、稳定的采样器 `MySampler`（继承 `SamplerBase`），通过“方差受控”的重要性采样流程估计固定配置下的 BER。

## 2. 模型与固定配置

- 信道：AWGN（加性高斯白噪声）。
- 码：Hamming 码，参数 `r=7`（码长 `n=127`，信息位 `k=120`）。
- 译码器：`chase(t=3)`，由评测器固定（不允许更改译码器与其参数）。
- 评测入口：评测器固定调用 `MySampler.simulate_variance_controlled(...)`，不使用普通 `simulate(...)` 路径。

## 3. 提交接口（程序级协议）

参赛者提交一个 Python 文件。评测器会动态加载该文件，并要求其中包含：

1. `class MySampler(SamplerBase)`；
2. `MySampler.simulate_variance_controlled(...)` 方法：内部执行仿真并返回结果。

你可以自行组织实现细节，但必须满足评测器对返回值形状与数值合法性的校验。

推荐返回格式与 `HammingCode.simulate_variance_controlled` 一致：

`(errors_log, weights_log, err_ratio, total_samples, actual_std, converged)`

返回值含义（与 `runtime/code_linear.py` 约定一致）：

- `errors_log`：错误事件计数的对数（log-domain）。
- `weights_log`：重要性权重和的对数（log-domain）。
- `err_ratio`：抽样中观测到的错误比例（非重要性加权真值）。
- `total_samples`：实际使用的总样本数。
- `actual_std`：估计过程的实际标准差。
- `converged`：是否达到 `target_std` 或满足评测器的收敛条件。

常用的 BER 对数估计为：

`err_rate_log = errors_log - weights_log`

## 4. 输入与输出（文件 I/O 协议）

评测器内部会构造并冻结输入常量（见下文）。为了避免题面歧义，这里给出推荐的“文件式协议”，便于离线复现与自测时对齐字段：

### 4.1 输入（推荐 `input.json`，评测器内部等价配置）

- `code_type`: 固定为 `"hamming"`
- `r`: 固定为 `7`
- `decoder`: 固定为 `"chase"`，并使用 `t=3`
- `sampler`: 由选手实现决定（评测器不直接读取该字段）
- `sigma`: 冻结噪声标准差
- `target_std`: 目标标准差（方差控制）
- `max_samples`: 最大样本数上限
- `batch_size`: 批大小
- `fix_tx`: 固定发射码字（评测器内部固定）
- `min_errors`: 最小错误事件数门槛（评测器内部固定）

### 4.2 输出（推荐 `results.json`，评测器内部等价提取）

必需：

- `err_rate_log`（建议由 `errors_log - weights_log` 得到）
- `err_ratio`
- `exec_time`

方差控制额外输出：

- `actual_samples`
- `actual_std`
- `converged`

说明：当前仓库的评测实现是“函数返回值协议”，而非写入 `results.json`。以上文件式协议用于题面清晰与复现实验时的字段对齐，等价于评测器从返回值中提取同名指标。

## 5. 评分与失败条件

设：

- `r`：选手估计的 BER（线性域，需为正且有限）。评测器会从 `err_rate_log` 计算 `r = exp(err_rate_log)` 或从等价返回值推导。
- `r0`：参考 BER（离线标定常量）。
- `t`：选手耗时（秒）。
- `t0`：基线耗时（常量）。
- `e = |log(r / r0)|`：相对误差的对数度量。

评分规则：

- 有效性约束：`e < epsilon`，否则 `combined_score = 0`。
- 综合得分：`combined_score = t0 / (t * e + 1e-6)`，越大越好。

失败条件（任一满足即 0 分）：

- 接口缺失或签名不匹配（无法加载 `MySampler` 或其方法）。
- 返回值形状不符合约定，或包含 `NaN/Inf` 等非有限数值。
- 评测器执行失败或超时。

## 6. 当前冻结评测常量

以下常量由评测器冻结（以 `benchmarks/WirelessChannelSimulation/HighReliableSimulation/eval/evaluator.py` 为准）：

- `sigma = 0.268`
- `target_std = 0.05`
- `max_samples = 100000`
- `batch_size = 10000`
- `min_errors = 20`
- `r0 = 5.52431776694918e-07`
- `t0 = 0.18551087379455566`
- `epsilon = 0.8`
