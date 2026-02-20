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

你可以自行组织实现细节，但必须满足评测器对类型、调用签名与返回值的校验。

### 3.1 评测器实际调用方式（以 `eval/evaluator.py` 为准）

评测器对提交程序执行的核心流程可等价理解为：

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

其中 `code` 由评测器固定构造为 `HammingCode(r=7, decoder="binary")`，并设置 `ChaseDecoder(t=3)`。

### 3.2 构造函数与继承要求

- `MySampler` 必须是 `SamplerBase` 的子类（评测器有 `issubclass` 检查）。
- `MySampler` 构造函数必须能接收关键字参数 `code` 与 `seed`（评测器按此实例化）。
- 推荐直接复用 `runtime/sampler.py` 的基类签名与写法。

### 3.3 `simulate_variance_controlled` 接口要求

`runtime/sampler.py` 中该方法约定为 keyword-only 形式：

```python
def simulate_variance_controlled(
    self,
    *,
    code,
    sigma,
    target_std,
    max_samples,
    batch_size,
    fix_tx=True,
    min_errors=10,
):
    ...
```

推荐实现：直接调用 `code.simulate_variance_controlled(..., sampler=self, ...)`。

### 3.4 `sample` 接口要求（重要性采样）

当你把 `sampler=self` 传给 `code.simulate_variance_controlled(...)` 后，底层会调用：

```python
noise_samples, log_pdf_proposal = sampler.sample(
    noise_std * sqrt(scale_factor), tx_bin, batch_size, **kwargs
)
```

因此 `sample` 至少需要满足：

- 返回 `noise_samples`，形状 `(batch_size, n)`；
- 返回 `log_pdf_proposal`，形状 `(batch_size,)`，且与采样分布一致。

### 3.5 返回值协议

推荐返回格式与 `HammingCode.simulate_variance_controlled` 一致（6元组）：

`(errors_log, weights_log, err_ratio, total_samples, actual_std, converged)`

评测器也接受字典格式（包含同名键）。

返回值含义（与 `runtime/code_linear.py` 约定一致）：

- `errors_log`：错误事件计数的对数（log-domain）。
- `weights_log`：重要性权重和的对数（log-domain）。
- `err_ratio`：抽样中观测到的错误比例（非重要性加权真值）。
- `total_samples`：实际使用的总样本数。
- `actual_std`：估计过程的实际标准差。
- `converged`：是否达到 `target_std` 或满足评测器的收敛条件。

常用的 BER 对数估计为：

`err_rate_log = errors_log - weights_log`

### 3.6 参数语义补充（按评测调用语境）

- `sigma`：AWGN 噪声标准差，值越大噪声越强。
- `target_std`：方差控制阈值；达到后可提前停止采样。
- `max_samples`：总样本数上限。
- `batch_size`：每轮采样批大小，影响收敛颗粒度与运行时间。
- `fix_tx`：评测固定为 `True`，即发送端固定全零码字以增强复现性。
- `min_errors`：最小错误事件门槛，避免“错误太少导致方差虚低”。

### 3.7 最小实现骨架（推荐）

```python
class MySampler(SamplerBase):
    def __init__(self, code, *, seed=0):
        super().__init__(code=code, seed=seed)

    def simulate_variance_controlled(
        self, *, code, sigma, target_std, max_samples, batch_size, fix_tx=True, min_errors=10
    ):
        return code.simulate_variance_controlled(
            noise_std=sigma,
            target_std=target_std,
            max_samples=max_samples,
            sampler=self,
            batch_size=batch_size,
            fix_tx=fix_tx,
            min_errors=min_errors,
        )
```

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

## 7. 本地运行与验证

### 7.1 环境准备

在仓库根目录执行（建议 Python 3.10+）：

```bash
pip install numpy scipy
```

如需运行 `frontier_eval` CLI，再安装：

```bash
pip install omegaconf hydra-core
```

### 7.2 快速验证提交程序

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/init.py
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/baseline/solution.py
```

### 7.3 单独运行评测器

评测 `scripts/init.py`：

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
repo = Path('.').resolve()
eval_path = repo / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "eval" / "evaluator.py"
prog = repo / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "scripts" / "init.py"
spec = importlib.util.spec_from_file_location("hrs_eval", str(eval_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.evaluate(str(prog), repo_root=repo))
PY
```

评测 `baseline/solution.py` 时，只需把 `prog` 改为 baseline 路径。

## 8. 接入 frontier_eval（可选）

当前任务名为 `high_reliable_simulation`。可用以下命令做入口连通性检查：

```bash
python -m frontier_eval task=high_reliable_simulation algorithm.iterations=0
```
