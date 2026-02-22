# HighReliableSimulation

本任务导航文档。

## 目标

实现 `MySampler`（继承 `SamplerBase`），并提供 `simulate_variance_controlled(...)`，在固定评测配置下估计 AWGN 信道中 Hamming(127,120) 的 BER。

## 文件

- `Task.md`：任务协议与评分规则（英文）。
- `Task_zh-CN.md`：任务协议中文版。
- `scripts/init.py`：最小可运行示例。
- `baseline/solution.py`：基线实现。
- `runtime/`：任务运行组件。
- `eval/evaluator.py`：评测入口。

## 快速运行

在仓库根目录执行：

```bash
python - <<'PY'
import importlib.util
from pathlib import Path

repo = Path('.').resolve()
eval_path = repo / 'benchmarks' / 'WirelessChannelSimulation' / 'HighReliableSimulation' / 'eval' / 'evaluator.py'
program = repo / 'benchmarks' / 'WirelessChannelSimulation' / 'HighReliableSimulation' / 'scripts' / 'init.py'

spec = importlib.util.spec_from_file_location('hrs_eval', str(eval_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.evaluate(str(program), repo_root=repo))
PY
```
