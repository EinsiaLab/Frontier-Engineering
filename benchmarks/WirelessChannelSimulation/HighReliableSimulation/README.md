# HighReliableSimulation

本目录为 `WirelessChannelSimulation` 领域下的具体任务导航文档，用于说明文件结构、如何运行与快速开始。

任务目标：实现一个自定义采样器 `MySampler`（继承 `SamplerBase`），并提供 `simulate_variance_controlled(...)` 接口，在冻结配置下估计 Hamming(127,120) 码在 AWGN 信道中的 BER（极低误码率场景）。

## 目录结构

```text
benchmarks/WirelessChannelSimulation/HighReliableSimulation/
├── README.md                 # 本文件：导航与快速开始
├── Task.md                   # 任务详情：背景、模型、接口、输入输出、评分与失败条件
├── USAGE.md                  # 更完整的本地运行与标定说明（可选阅读）
├── calibration_final.json    # 冻结评测常量的来源记录（标定结果快照）
├── scripts/
│   ├── init.py               # 初始化提交示例程序（供本地快速验证）
│   └── calibrate_sigma.py    # 标定脚本（开发用途）
├── baseline/
│   └── solution.py           # 参考实现（baseline）
├── runtime/                  # 任务运行所需的最小依赖（与可靠仿真实现绑定）
│   ├── code_linear.py
│   ├── sampler.py
│   └── chase.py
└── eval/
    └── evaluator.py          # 评测入口（等价于仓库规范中的 verification/evaluator.py）
```

说明：仓库规范建议使用 `verification/evaluator.py` 作为评测入口。本任务将评测脚本放在 `eval/evaluator.py`，其角色与职责等价，文档中统一称为“评测器/评分脚本入口”。

## 快速开始

在仓库根目录运行一次评测器对示例程序打分：

```bash
python - <<'PY'
import importlib.util
from pathlib import Path

repo = Path(".").resolve()
eval_path = repo / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "eval" / "evaluator.py"
program = repo / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "scripts" / "init.py"

spec = importlib.util.spec_from_file_location("hrs_eval", str(eval_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.evaluate(str(program), repo_root=repo))
PY
```

如需更详细的运行与标定说明，见 `benchmarks/WirelessChannelSimulation/HighReliableSimulation/USAGE.md`。

## 冻结配置提示

- 评测入口固定调用 `MySampler.simulate_variance_controlled(...)`，不走普通 `simulate(...)`。
- 译码器固定为 `chase(t=3)`（由评测器写死）。
- 当前评测噪声与评分常量已冻结（见 `benchmarks/WirelessChannelSimulation/HighReliableSimulation/Task.md` 与 `benchmarks/WirelessChannelSimulation/HighReliableSimulation/calibration_final.json`）。
