# 本地运行指南（USAGE）

## 1. 环境准备

在仓库根目录执行，建议 Python 3.10+：

```bash
pip install numpy scipy
```

若需要运行 `frontier_eval` CLI，还需要：

```bash
pip install omegaconf hydra-core
```

说明：
- 本任务核心运行依赖已内置到 `runtime/`。

## 2. 快速验证任务代码

运行初始化示例（`BesselSampler`）：

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/init.py
```

运行 baseline（`BesselSampler`）：

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/baseline/solution.py
```

## 3. 单独运行评测器

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

评测 `baseline/solution.py`（仅修改 `prog` 路径）：

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
repo = Path('.').resolve()
eval_path = repo / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "eval" / "evaluator.py"
prog = repo / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "baseline" / "solution.py"
spec = importlib.util.spec_from_file_location("hrs_eval", str(eval_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.evaluate(str(prog), repo_root=repo))
PY
```

## 4. 冻结常量与评分说明

当前 `eval/evaluator.py` 冻结常量：

- `sigma = 0.268`
- `target_std = 0.05`
- `max_samples = 100000`
- `batch_size = 10000`
- `min_errors = 20`
- `r0 = 5.52431776694918e-07`
- `t0 = 0.18551087379455566`
- `epsilon = 0.8`

评分：

```text
e = |log(r / r0)|
if e < epsilon:
  combined_score = t0 / (t * e + 1e-6)
else:
  combined_score = 0
```

## 5. 如何提交你自己的采样器

提交文件需定义：

1. `class MySampler(SamplerBase)`
2. `simulate_variance_controlled(...)` 方法

## 5.1 接口与函数定义（重要）

以下接口以当前仓库代码为准，主要参考：

- 抽样器基类：`benchmarks/WirelessChannelSimulation/HighReliableSimulation/runtime/sampler.py`
- 汉明码实现：`benchmarks/WirelessChannelSimulation/HighReliableSimulation/runtime/code_linear.py`
- chase译码器实现：`benchmarks/WirelessChannelSimulation/HighReliableSimulation/runtime/chase.py`
- 评测器调用方式：`benchmarks/WirelessChannelSimulation/HighReliableSimulation/eval/evaluator.py`

### 5.1.1 必需的类与构造函数

评测器会动态加载你的提交文件，并执行（等价）：

```python
sampler = MySampler(code=code)
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

参数说明：

- `code`：评测器构造的编码对象（`HammingCode(r=7)`），并已设置译码器为 `ChaseDecoder(t=3)`；你应当把它视为只读依赖，用来调用 `code.simulate_variance_controlled(...)` 或获取码参数（如 `code.dim`）。
- `seed`：本次评测重复（repeat）的随机种子。评测器会用它初始化 `code.rng`，并在可能时覆盖 `sampler.rng`，用于复现实验与降低随机波动。
- `sigma`：AWGN 噪声标准差（noise std）。`sigma` 越大信道越“吵”，BER 越高。该值在评测中被冻结为常量（见 `Task.md`）。
- `target_std`：方差控制的目标标准差阈值。仿真会按 batch 迭代，当累计满足最小批次数与最小错误事件数门槛后，若估计标准差低于该阈值则可提前停止（节省时间）。
- `max_samples`：总样本数上限（噪声样本条数）。无论是否收敛，总采样量不会超过该值。
- `batch_size`：每个 batch 的样本数。仿真会重复生成多个 batch，并根据收敛条件决定何时停止；`batch_size` 也会影响每轮更新标准差的颗粒度与运行时间/内存峰值。
- `fix_tx`：是否固定发送码字。评测器写死为 `True`，表示每个 batch 的发送比特固定为全零码字（常见做法，用于降低随机性并提升复现性）；若为 `False` 则每轮会随机生成发送比特（本任务评测不走该路径）。
- `min_errors`：最小错误事件数门槛。只有当累计观测到的错误事件数达到该值后，才会开始计算/判断标准差是否低于 `target_std`，避免“几乎没错误”导致的虚假低方差。

因此，`MySampler` 必须满足：

- 必须继承 `SamplerBase`；
- 构造函数需要能接收关键字参数 `code` 与 `seed`（至少不能因为这两个参数报错）。

建议的最小形式：

```python
class MySampler(SamplerBase):
    def __init__(self, code, seed=0):
        super().__init__()
        self.code = code
```

### 5.1.2 `simulate_variance_controlled`（评测入口）

`SamplerBase` 中该方法是关键字参数（keyword-only）形式，签名为：

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

推荐实现方式：直接调用 `code.simulate_variance_controlled(...)`，并把 `sampler=self` 传入（见 `runtime/sampler.py` 中的 `NaiveSampler` / `BesselSampler` 示例）。

该方法的推荐返回值格式为 6 元组：

`(errors_log, weights_log, err_ratio, total_samples, actual_std, converged)`

返回值含义：

- `errors_log`：重要性加权后的“错误权重和”的对数（log-domain）。
- `weights_log`：重要性权重和的对数（log-domain）。
- `err_ratio`：在采样中观测到的错误比例（未做重要性加权，仅用于诊断）。
- `total_samples`：实际使用的样本数。
- `actual_std`：估计过程的实际标准差（评测中会记录该指标）。
- `converged`：布尔值，表示是否在 `max_samples` 之前达到 `target_std`（或等价收敛条件）。

通常 BER 的对数估计为：

`err_rate_log = errors_log - weights_log`

### 5.1.3 `sample`（重要性采样分布）

当你通过 `code.simulate_variance_controlled(..., sampler=self, ...)` 接入仿真时，仿真会调用：

```python
noise_samples, log_pdf_proposal = sampler.sample(
    noise_std * sqrt(scale_factor),
    tx_bin,
    batch_size,
    **kwargs,
)
```

因此你的 `sample` 必须返回：

- `noise_samples`：形状为 `(batch_size, n)` 的噪声样本数组（`n=127`）。
- `log_pdf_proposal`：形状为 `(batch_size,)` 的对数密度 `log q(noise_samples)`。

注意：`log_pdf_proposal` 必须与 `noise_samples` 对应的提议分布一致，否则会导致 `log_weights = log p - log q` 失真，评分会明显下降或直接失效。

推荐做法：继承 `SamplerBase`（或其子类），并在 `simulate_variance_controlled` 中调用：

```python
code.simulate_variance_controlled(
    noise_std=sigma,
    target_std=target_std,
    max_samples=max_samples,
    sampler=self,
    batch_size=batch_size,
    fix_tx=True,
    min_errors=min_errors,
)
```

## 6. 接入 frontier_eval（可选）

当前已注册任务名：`high_reliable_simulation`。

如果你已安装 `frontier_eval` 依赖，可运行：

```bash
python -m frontier_eval task=high_reliable_simulation algorithm.iterations=0
```

`iterations=0` 仅用于验证任务加载与评测入口，不进行演化。
