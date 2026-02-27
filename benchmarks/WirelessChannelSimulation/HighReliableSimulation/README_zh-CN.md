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
- `verification/evaluator.py`：评测入口。
- `verification/requirements.txt`：本地运行评测器的最小依赖。

## 环境配置

在仓库根目录执行：

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/WirelessChannelSimulation/HighReliableSimulation/verification/requirements.txt
```

## 快速运行

在仓库根目录执行：

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/verification/evaluator.py benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/init.py
```

或在任务目录执行：

```bash
cd benchmarks/WirelessChannelSimulation/HighReliableSimulation && python verification/evaluator.py scripts/init.py
```

`scripts/init.py` 是可运行初始程序；在正常环境下应出现非零 `runtime_s`，且 `valid=1.0`。

## frontier_eval 任务名

该任务注册的 `task_name` 为：

```text
high_reliable_simulation
```

示例：

```bash
python -m frontier_eval task=high_reliable_simulation algorithm.iterations=0
```
