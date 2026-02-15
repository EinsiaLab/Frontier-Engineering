# HighReliableSimulation

该任务要求参赛者实现一个 `MySampler` 类（继承 `SamplerBase`），并提供
`simulate_variance_controlled` 接口，在固定配置下估计 Hamming(127,120) 的 BER。

## 文件结构

```text
HighReliableSimulation/
├── README.md
├── Task.md
├── USAGE.md
├── IMPLEMENTATION_PLAN.md
├── calibration.md
├── calibration_final.json
├── scripts/
│   ├── init.py
│   └── calibrate_sigma.py
├── baseline/
│   └── solution.py
└── eval/
    └── evaluator.py
```

## 快速测试

在仓库根目录运行：

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
repo = Path('.').resolve()
eval_path = repo / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "eval" / "evaluator.py"
program = repo / "benchmarks" / "WirelessChannelSimulation" / "HighReliableSimulation" / "scripts" / "init.py"
spec = importlib.util.spec_from_file_location("hrs_eval", str(eval_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(mod.evaluate(str(program), repo_root=repo))
PY
```

## 当前阶段说明

- 当前评测常量已冻结为 `sigma=0.268`（见 `Task.md` 与 `calibration.md`）。
- 标定脚本：`scripts/calibrate_sigma.py`
- 标定记录：`calibration.md`、`calibration_final.json`

## 详细用法

完整用法见：`benchmarks/WirelessChannelSimulation/HighReliableSimulation/USAGE.md`
