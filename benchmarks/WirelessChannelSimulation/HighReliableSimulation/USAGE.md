# USAGE

## 1. 环境准备

在仓库根目录执行，建议 Python 3.10+：

```bash
pip install numpy scipy
```

若需要跑 `frontier_eval` CLI，还需要：

```bash
pip install omegaconf hydra-core
```

## 2. 快速验证任务代码

运行初始提交（`NaiveSampler`）：

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/init.py
```

运行 baseline（`BesselSampler`）：

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/baseline/solution.py
```

## 3. 单独运行评测器

评测 `init.py`：

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

评测 `baseline/solution.py`（只改 `prog` 路径）：

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

## 4. 评测常量说明

当前 `evaluator.py` 固定常量：

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

## 6. 重新标定 sigma / r0 / t0

运行标定脚本（示例）：

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/calibrate_sigma.py \
  --sigmas 0.26 0.264 0.268 0.272 0.276 \
  --target-ber 1e-7 \
  --target-std 0.05 \
  --max-samples 100000 \
  --batch-size 10000 \
  --min-errors 20 \
  --repeats 5 \
  --output benchmarks/WirelessChannelSimulation/HighReliableSimulation/calibration_new.json
```

然后把 `selected` 中的值同步到 `eval/evaluator.py`：

- `sigma_star -> DEV_SIGMA`
- `r0 -> R0_DEV`
- `t0 -> T0_DEV`

## 7. 接入 frontier_eval

当前已注册任务名：`high_reliable_simulation`。

如果你已安装 `frontier_eval` 依赖，可运行：

```bash
python -m frontier_eval task=high_reliable_simulation algorithm.iterations=0
```

`iterations=0` 仅用于验证任务加载与评测入口，不进行演化。

