# HighReliableSimulation

Navigation document for this task.

## Goal

Implement `MySampler` (inherits `SamplerBase`) and provide `simulate_variance_controlled(...)` to estimate BER for Hamming(127,120) over AWGN under fixed evaluator settings.

## Files

- `Task.md`: task contract and scoring rules (English).
- `Task_zh-CN.md`: Chinese version of task contract.
- `scripts/init.py`: minimal runnable starter.
- `baseline/solution.py`: baseline implementation.
- `runtime/`: task runtime components.
- `verification/evaluator.py`: evaluator entry.
- `verification/requirements.txt`: minimal dependencies for local evaluator run.

## Environment

From repository root:

```bash
pip install -r frontier_eval/requirements.txt
pip install -r benchmarks/WirelessChannelSimulation/HighReliableSimulation/verification/requirements.txt
```

## Quick Run

Run from repository root:

```bash
python benchmarks/WirelessChannelSimulation/HighReliableSimulation/verification/evaluator.py benchmarks/WirelessChannelSimulation/HighReliableSimulation/scripts/init.py
```

Or run from the task directory:

```bash
cd benchmarks/WirelessChannelSimulation/HighReliableSimulation && python verification/evaluator.py scripts/init.py
```

`scripts/init.py` is expected to be runnable and should produce non-zero `runtime_s` and `valid=1.0` under a normal environment.

## frontier_eval Task Name

Registered `task_name` for this task:

```text
high_reliable_simulation
```

Example:

```bash
python -m frontier_eval task=high_reliable_simulation algorithm.iterations=0
```
